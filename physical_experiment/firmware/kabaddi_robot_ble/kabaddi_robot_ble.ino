/*
 * =============================================================================
 *  RoboKabaddi — ONE sketch for every robot. ESP32 BUILT-IN WiFi, no FSR.
 * =============================================================================
 *
 *  Replaces kabaddi_raider.ino + kabaddi_defender.ino.
 *  Set ROBOT_ROLE and ROBOT_INDEX below, flash, done.
 *
 *  vs the ESP8266 version
 *  ----------------------
 *   - external ESP8266 AT bridge   -> ESP32 built-in WiFi (WiFiClient)
 *   - "+IPD,N:CMD" parsing         -> client.read() gives the raw byte
 *   - blocking WiFi join in setup  -> non-blocking state machine in loop()
 *   - FSR force sensing            -> REMOVED (see below)
 *   - BLE                          -> off by default (see below)
 *
 *  NO FSR. Contact for the ICRA trials is detected from ArUco tracking
 *  (CONTACT_DIST_M in hw_experiment.py), not from force sensors, so all the
 *  ADC sampling, debounce and contact-event logic is gone. The robot still
 *  emits one CSV line per second in the ORIGINAL column layout with the force
 *  fields zeroed, because fsr_listener.py is also the TCP server and parses
 *  that format. Keeping the heartbeat means:
 *    - fsr_listener.py needs no changes at all;
 *    - the robot notices a dead link within ~1 s via its own write failure,
 *      instead of sitting on a half-open socket.
 *
 *  WATCHDOG (TG1WDT_SYS_RESET). The radio draws a ~300 mA burst on its first
 *  association frame; on this board that trips the Timer-Group-1 watchdog,
 *  which is why the Paper 1 sketch disabled the built-in radio entirely.
 *  Three mitigations here: TX power capped, WiFi started before the motors
 *  (supply idle), and setup() never blocks. Note TG1WDT is the INTERRUPT
 *  watchdog — esp_task_wdt_reset() does not apply to it and is not used.
 *
 *  BLE. BLE and WiFi share one radio; running both is what caused the resets
 *  the ESP8266 was added to avoid. The autonomous experiment does not need
 *  BLE, so it is compiled out. Set ENABLE_BLE 1 only for manual testing.
 *
 *  PC SIDE IS UNCHANGED. robot_controller.py and fsr_listener.py work as-is.
 * =============================================================================
 */

// ============================ ROLE SELECTION =================================
#define _STR(x) #x
#define STR(x) _STR(x)

//  0 = defender   1 = raider
#define ROBOT_ROLE    0
#define ROBOT_INDEX   1        // 1/2/3 for defenders; ignored for the raider

#if ROBOT_ROLE == 1
  #define ROBOT_ID      "raider_1"
  #define BLE_NAME      "Smorphi_Raider1"
  #define PC_PORT_NUM   5000                     // fsr_listener.ROBOT_PORTS
#else
  #define ROBOT_ID      "defender_" STR(ROBOT_INDEX)
  #define BLE_NAME      "Smorphi_Defender" STR(ROBOT_INDEX)
  #define PC_PORT_NUM   (5000 + ROBOT_INDEX)     // 5001 / 5002 / 5003
#endif

// ============================ NETWORK ========================================
const char* WIFI_SSID_PRIMARY = "YOUR_WIFI_SSID";
const char* WIFI_PASS_PRIMARY = "YOUR_WIFI_PASSWORD";
const char* WIFI_SSID_BACKUP  = "YOUR_WIFI_SSID";      // lab network
const char* WIFI_PASS_BACKUP  = "";

const char* PC_IP_PRIMARY = "192.168.1.100";      // Mac IP on YOUR_NETWORK
const char* PC_IP_BACKUP  = "192.168.1.100";      // <-- set to the Mac IP on the LAB net
const uint16_t PC_PORT    = PC_PORT_NUM;           // set by the role block above

#define WIFI_JOIN_TIMEOUT_MS  15000
#define TCP_RETRY_MS          5000
#define HEARTBEAT_MS          1000   // status line to the PC at 1 Hz
#define LED_PIN                  2   // solid = TCP up, blink = connecting

// ---- watchdog / brownout mitigation (TG1WDT_SYS_RESET) ---------------------
// (WIFI_TX_POWER is defined below, inside the USE_WIFI guard: the symbol comes
//  from WiFi.h, which is only included in WiFi mode.)

// Set 1 to skip Smorphi motor init. DIAGNOSTIC ONLY: if WiFi connects with
// this at 1 and resets at 0, the fault is the motor board (power or timer
// conflict), not the WiFi code.
#define DIAGNOSTIC_NO_MOTORS 0

// ---- TRANSPORT -------------------------------------------------------------
// USE_WIFI 1 + ENABLE_BLE 0 : built-in WiFi/TCP        (host: --link wifi)
// USE_WIFI 0 + ENABLE_BLE 1 : BLE only                 (host: --link ble)
//
// BLE-only is the Paper 1 radio configuration minus the ESP8266, and it is the
// known-good path on this board: WiFi and BLE share one radio, and the WiFi
// association burst is what trips TG1WDT. Never set both to 1.
#define USE_WIFI    0      // <-- BLE mode
#define ENABLE_BLE  1      // <-- BLE mode

// WiFi phases. Plain #defines, not an enum: the Arduino IDE hoists generated
// prototypes above the sketch body, so a custom type in a signature fails to
// compile ("'WifiPhase' has not been declared").
#define W_IDLE          0
#define W_JOIN_PRIMARY  1
#define W_JOIN_BACKUP   2
#define W_UP            3
#define W_FAILED        4

// ============================ INCLUDES =======================================
// Only ONE radio stack is compiled. Linking both overflows the default 1.3 MB
// app partition ("text section exceeds available space in board").
#include <smorphi_single.h>

#if USE_WIFI
  #include <WiFi.h>
  #define WIFI_TX_POWER WIFI_POWER_8_5dBm   // halves the burst that trips TG1WDT
#endif

#if ENABLE_BLE
  #include <BLEDevice.h>
  #include <BLEServer.h>
  #include <BLEUtils.h>
  #include <BLE2902.h>
  #define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
  #define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
  #define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
  BLECharacteristic *pCharacteristic;
#endif

// ============================ STATE ==========================================
Smorphi_single my_robot;
#if USE_WIFI
WiFiClient client;
#endif

unsigned long lastHeartbeat = 0, lastTcpRetry = 0;
unsigned long experimentStartTime = 0, wifiAttemptStart = 0, lastBlink = 0;
int  wifiPhase = W_IDLE;
bool wifiReady = false;
const char* activePC_IP = nullptr;
unsigned long cmdCount = 0;

// ============================ COMMAND HANDLER ================================
// Unchanged ASCII map — matches robot_controller.py / smorphi_controller.py.
void handleCommand(String d)
{
    cmdCount++;
#if DIAGNOSTIC_NO_MOTORS
    Serial.println("[DIAG] cmd " + d + " (motors disabled)");
    return;
#else
    if (d == "[")       { my_robot.MoveForward(100); }
    else if (d == "A")  { my_robot.MoveForward(75); }
    else if (d == "]")  { my_robot.MoveForward(50); }
    else if (d == "^")  { my_robot.MoveForward(25); }
    else if (d == "_")  { my_robot.MoveBackward(100); }
    else if (d == "`")  { my_robot.MoveBackward(75); }
    else if (d == "a")  { my_robot.MoveBackward(50); }
    else if (d == "b")  { my_robot.MoveBackward(25); }
    else if (d == "c")  { my_robot.MoveLeft(100); }
    else if (d == "d")  { my_robot.MoveLeft(75); }
    else if (d == "e")  { my_robot.MoveLeft(50); }
    else if (d == "f")  { my_robot.MoveLeft(25); }
    else if (d == "g")  { my_robot.MoveRight(100); }
    else if (d == "h")  { my_robot.MoveRight(75); }
    else if (d == "i")  { my_robot.MoveRight(50); }
    else if (d == "j")  { my_robot.MoveRight(25); }
    else if (d == "k")  { my_robot.MoveDiagUpLeft(100); }
    else if (d == "l")  { my_robot.MoveDiagUpLeft(75); }
    else if (d == "m")  { my_robot.MoveDiagUpLeft(50); }
    else if (d == "n")  { my_robot.MoveDiagUpLeft(25); }
    else if (d == "o")  { my_robot.MoveDiagUpRight(100); }
    else if (d == "p")  { my_robot.MoveDiagUpRight(75); }
    else if (d == "q")  { my_robot.MoveDiagUpRight(50); }
    else if (d == "r")  { my_robot.MoveDiagUpRight(25); }
    else if (d == "s")  { my_robot.MoveDiagDownLeft(100); }
    else if (d == "t")  { my_robot.MoveDiagDownLeft(75); }
    else if (d == "u")  { my_robot.MoveDiagDownLeft(50); }
    else if (d == "v")  { my_robot.MoveDiagDownLeft(25); }
    else if (d == "w")  { my_robot.MoveDiagDownRight(100); }
    else if (d == "x")  { my_robot.MoveDiagDownRight(75); }
    else if (d == "y")  { my_robot.MoveDiagDownRight(50); }
    else if (d == "z")  { my_robot.MoveDiagDownRight(25); }
    else if (d == "{")  { my_robot.CenterPivotLeft(270); }
    else if (d == "|")  { my_robot.CenterPivotRight(270); }
    else if (d == "}")  { my_robot.stopSmorphi_single(); }
    my_robot.sm_reset_M1();
#endif
}

#if ENABLE_BLE
class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *s)    {}
    void onDisconnect(BLEServer *s) { s->getAdvertising()->start(); }
};
class MyCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *c) {
        String v = c->getValue().c_str();
        if (v.length() > 0) handleCommand(v);
    }
};
#endif

// ============================ WIFI / TCP =====================================
#if USE_WIFI
void beginJoin(const char* ssid, const char* pass, int phase) {
    Serial.printf("[WiFi] joining %s ...\n", ssid);
    WiFi.disconnect(true, true);
    delay(50);
    WiFi.begin(ssid, pass);
    wifiPhase = phase;
    wifiAttemptStart = millis();
}

void startWiFi() {
    WiFi.persistent(false);         // don't write flash on every begin()
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);           // power-save would add 100-300 ms latency
    WiFi.setTxPower(WIFI_TX_POWER); // limits the burst that trips TG1WDT
    WiFi.setAutoReconnect(true);
    beginJoin(WIFI_SSID_PRIMARY, WIFI_PASS_PRIMARY, W_JOIN_PRIMARY);
}

bool serviceWiFi() {
    if (wifiPhase == W_UP) {
        if (WiFi.status() == WL_CONNECTED) return true;
        Serial.println("[WiFi] link lost");
        wifiReady = false;
        beginJoin(WIFI_SSID_PRIMARY, WIFI_PASS_PRIMARY, W_JOIN_PRIMARY);
        return false;
    }
    if (wifiPhase == W_JOIN_PRIMARY || wifiPhase == W_JOIN_BACKUP) {
        if (WiFi.status() == WL_CONNECTED) {
            activePC_IP = (wifiPhase == W_JOIN_PRIMARY) ? PC_IP_PRIMARY : PC_IP_BACKUP;
            wifiPhase = W_UP;
            wifiReady = true;
            Serial.printf("[WiFi] connected. robot IP %s  ->  PC %s:%u\n",
                          WiFi.localIP().toString().c_str(), activePC_IP, PC_PORT);
            return true;
        }
        if (millis() - wifiAttemptStart > WIFI_JOIN_TIMEOUT_MS) {
            if (wifiPhase == W_JOIN_PRIMARY) {
                Serial.println("[WiFi] primary timed out");
                beginJoin(WIFI_SSID_BACKUP, WIFI_PASS_BACKUP, W_JOIN_BACKUP);
            } else {
                Serial.println("[WiFi] both networks failed — retrying in 10 s");
                wifiPhase = W_FAILED;
                wifiAttemptStart = millis();
            }
        }
    }
    if (wifiPhase == W_FAILED && millis() - wifiAttemptStart > 10000) {
        beginJoin(WIFI_SSID_PRIMARY, WIFI_PASS_PRIMARY, W_JOIN_PRIMARY);
    }
    return false;
}

bool connectTCP() {
    if (!wifiReady) return false;
    Serial.printf("[TCP] connecting to %s:%u ...\n", activePC_IP, PC_PORT);
    if (!client.connect(activePC_IP, PC_PORT)) {
        Serial.println("[TCP] failed");
        return false;
    }
    client.setNoDelay(true);        // no Nagle — command bytes go out at once
    Serial.println("[TCP] connected");
    // Same header fsr_listener.py has always received.
    client.println("timestamp_ms,robot_id,left_adc,left_force_g,left_peak_g,"
                   "right_adc,right_force_g,right_peak_g,"
                   "left_contact,right_contact,event,elapsed_ms");
    return true;
}

#endif  // USE_WIFI

// One CSV line in the ORIGINAL column layout, force fields zeroed.
// Keeps fsr_listener.py's parser happy and proves the link is alive.
void sendHeartbeat(unsigned long now) {
    char buf[192];
    snprintf(buf, sizeof(buf),
             "%lu,%s,0,0.0,0.0,0,0.0,0.0,0,0,STREAM,%lu",
             now, ROBOT_ID, now - experimentStartTime);
#if USE_WIFI
    if (client.connected()) client.println(buf);
#endif
    Serial.printf("%s   cmds=%lu\n", buf, cmdCount);
}

// ============================ SETUP ==========================================
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n==========================================");
    Serial.printf("  RoboKabaddi — %s  (built-in WiFi, no FSR)\n", ROBOT_ID);
    Serial.println("==========================================\n");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // WiFi first, asynchronously: the motors are still idle so the supply has
    // headroom for the association burst, and setup() never blocks.
#if USE_WIFI
    startWiFi();
    Serial.println("[OK] WiFi association started (non-blocking)");
#else
    // WiFi.h is not even compiled in; the radio is simply never started.
    Serial.println("[OK] BLE only (WiFi not compiled in)");
#endif

#if DIAGNOSTIC_NO_MOTORS
    Serial.println("[DIAG] Smorphi init SKIPPED");
#else
    my_robot.BeginSmorphi_single();
    Serial.println("[OK] Smorphi initialized");
#endif

#if ENABLE_BLE
    BLEDevice::init(BLE_NAME);
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID_TX, BLECharacteristic::PROPERTY_NOTIFY);
    pCharacteristic->addDescriptor(new BLE2902());
    BLECharacteristic *pRx = pService->createCharacteristic(
        CHARACTERISTIC_UUID_RX, BLECharacteristic::PROPERTY_WRITE);
    pRx->setCallbacks(new MyCallbacks());
    pService->start();
    pServer->getAdvertising()->start();
    Serial.println("[OK] BLE started (WiFi latency will be worse)");
#endif

    Serial.println("[INFO] waiting for WiFi + TCP in loop() ...");
    experimentStartTime = millis();
    lastHeartbeat = millis();
}

// ============================ LOOP ===========================================
void loop() {
    unsigned long now = millis();

#if USE_WIFI
    // 1. WiFi state machine
    //    No esp_task_wdt_reset() here: the Arduino loopTask is not subscribed
    //    to the Task WDT, so the call only logs "task not found" every pass.
    //    It would not have helped regardless — the reset we were chasing is
    //    TG1WDT, the INTERRUPT watchdog. What prevents that is a setup() that
    //    never blocks plus the reduced TX power, both of which are in place.
    bool up = serviceWiFi();

    // 2. TCP connect / reconnect
    if (up && !client.connected() && now - lastTcpRetry > TCP_RETRY_MS) {
        lastTcpRetry = now;
        client.stop();
        connectTCP();
    }

    // 3. commands from the PC — raw bytes, no "+IPD," wrapper to strip.
    //    One ASCII char per command; \r and \n ignored, so this works whether
    //    or not the PC terminates its writes with a newline.
    while (client.available()) {
        char c = client.read();
        if (c == '\r' || c == '\n') continue;
        handleCommand(String(c));
    }

#endif  // USE_WIFI

    // 4. heartbeat (serial always; TCP only when the WiFi link is up)
    if (now - lastHeartbeat >= HEARTBEAT_MS) {
        lastHeartbeat = now;
        sendHeartbeat(now);
    }

    // 5. LED: solid when linked, blinking while connecting
#if USE_WIFI
    if (client.connected()) {
#else
    if (cmdCount > 0) {          // BLE: solid once the PC has sent anything
#endif
        digitalWrite(LED_PIN, HIGH);
    } else if (now - lastBlink > 250) {
        lastBlink = now;
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    }

    delay(1);
}
