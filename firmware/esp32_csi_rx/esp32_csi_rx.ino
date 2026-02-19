#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

// ---------------------------------------------------------------------------------
// CONFIGURATION
// ---------------------------------------------------------------------------------
#define WIFI_SSID "CSI_NET"       // <--- SET YOUR WIFI CREDENTIALS HERE
#define WIFI_PASS "1234567890"    // <--- SET YOUR WIFI CREDENTIALS HERE
// #define TARGET_IP "255.255.255.255" // REMOVED: Using dynamic Gateway IP
#define TARGET_PORT 8888          // Default port for our Python script
IPAddress target_ip;


WiFiUDP udp;
// Buffer for CSI data to send via UDP
// Format: 1 byte len, 4 bytes timestamp, 1 byte rssi, then payload
uint8_t buffer[1024]; 

void _csi_cb(void *ctx, wifi_csi_info_t *data) {
    wifi_csi_info_t *d = (wifi_csi_info_t *)data;
    
    // Filter: Only interested in our AP or specific MACs if needed
    // For now, capture everything to ensure data flow
    
    int len = d->len; // Total length of CSI data
    if (len > 4096) return; // Safety check for buffer
    
    // Construct Packet
    // [Header: "CSI" (3), Seq (4), Len (2)] + [Payload]
    // RSSI REMOVED as per requirements
    
    uint32_t timestamp = millis();
    
    // Packet Structure (Custom binary protocol for speed)
    // 0-2: "CSI"
    // 3-6: Timestamp
    // 7-8: Data Length (uint16)
    // 9...: Raw CSI Data (int8 arrays of imag/real)
    
    buffer[0] = 'C'; buffer[1] = 'S'; buffer[2] = 'I';
    memcpy(&buffer[3], &timestamp, 4);
    uint16_t data_len = (uint16_t)len;
    memcpy(&buffer[7], &data_len, 2);
    
    if (d->buf != NULL) {
        memcpy(&buffer[9], d->buf, len);
        // Send UDP
        // Use the global target IP set in setup()
        udp.beginPacket(target_ip, TARGET_PORT);
        udp.write(buffer, 9 + len);
        udp.endPacket();
    }
}

void setup() {
    Serial.begin(115200);
    
    // 1. Connect to Wi-Fi
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    Serial.print("Connecting to ");
    Serial.println(WIFI_SSID);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    // AUTOMATIC GATEWAY DETECTION (For Hotspot/Direct Connection)
    target_ip = WiFi.gatewayIP();
    Serial.print("Target IP (Gateway): ");
    Serial.println(target_ip);


    // 2. Configure CSI
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(_csi_cb, NULL));
    
    // Enable CSI for all packets (Beacon, Data, etc generally available)
    wifi_csi_config_t configuration_csi;
    configuration_csi.lltf_en = 1;
    configuration_csi.htltf_en = 1;
    configuration_csi.stbc_htltf2_en = 1;
    configuration_csi.ltf_merge_en = 1;
    configuration_csi.channel_filter_en = 1;
    configuration_csi.manu_scale = 0;
    configuration_csi.shift = 0;
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&configuration_csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    
    Serial.println("CSI Capture Started");
}

void loop() {
    // Keep alive or print stats
    delay(1000);
}
