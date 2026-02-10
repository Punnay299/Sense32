#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

// ================= COFIGURATION =================
#define WIFI_SSID "CSI_NET"
#define WIFI_PASS "1234567890"
#define TARGET_IP "10.42.0.1"
 // Laptop IP (AP Gateway usually)
#define TARGET_PORT 8888
// ================================================

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
    // [Header: "CSI" (3), Seq (4), RSSI (1), Len (2)] + [Payload]
    // RSSI is in d->rx_ctrl.rssi
    
    int8_t rssi = d->rx_ctrl.rssi;
    uint32_t timestamp = millis();
    
    // Packet Structure (Custom binary protocol for speed)
    // 0-2: "CSI"
    // 3: RSSI (signed int8 cast to uint8)
    // 4-7: Timestamp
    // 8-9: Data Length (uint16)
    // 10...: Raw CSI Data (int8 arrays of imag/real)
    
    buffer[0] = 'C'; buffer[1] = 'S'; buffer[2] = 'I';
    buffer[3] = (uint8_t)rssi;
    memcpy(&buffer[4], &timestamp, 4);
    uint16_t data_len = (uint16_t)len;
    memcpy(&buffer[8], &data_len, 2);
    
    if (d->buf != NULL) {
        memcpy(&buffer[10], d->buf, len);
        // Send UDP
        udp.beginPacket(TARGET_IP, TARGET_PORT);
        udp.write(buffer, 10 + len);
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
