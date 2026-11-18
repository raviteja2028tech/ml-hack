# Two-Laptop Demo Guide (Windows)

**Laptop 1** runs the FastAPI backend, which holds the saved LightGBM model and the HMM. **Laptop 2** runs the dashboard. Both must be on the same Wi-Fi or LAN.

## 1. Laptop 1: start the backend
```powershell
cd C:\Users\ravit\OneDrive\Desktop\ml-hackthon
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Wait for `[PathSnap] model loaded ... 10 student-test trips` and `Uvicorn running on http://0.0.0.0:8000`.

Check it locally: `curl http://localhost:8000/health` should report `"model_loaded": true`.

## 2. Laptop 1: find its LAN IP
```powershell
ipconfig | findstr IPv4
# or
Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -notlike '127.*' | Select IPAddress, InterfaceAlias
```
Use the Wi-Fi/Ethernet address, for example `10.5.18.34`. Skip virtual adapters such as `192.168.56.x` (VirtualBox).

## 3. Laptop 1: allow port 8000 through the firewall (run as Administrator)
```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Domain
```
- The Wi-Fi network must be set to **Private**, not Public, or the rule must include `-Profile Public`.
- If Windows shows a "Python wants to access the network" pop-up, click **Allow**.
- To remove the rule afterwards: `Remove-NetFirewallRule -DisplayName "PathSnap API 8000"`.

## 4. Laptop 2: open the dashboard
**Option A** needs nothing installed on Laptop 2. Open `http://<LAPTOP1_IP>:8000/ui/` in a browser.

**Option B** serves the UI from Laptop 2. Copy the `src/phase8_ui` folder to Laptop 2, then:
```powershell
python -m http.server 5500 --directory phase8_ui
```
Open `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000`.

You can also type the backend URL into the box at the top right and press **Connect**; it is remembered. CORS is open by default. To restrict it, start the backend with `$env:PATHSNAP_CORS_ORIGINS="http://<LAPTOP2_IP>:5500"`.

## 5. Demo script
1. The status pill reads "Connected · lightgbm + HMM + Viterbi".
2. Choose trip **00000052** (262 points, about 0.2 s) and press **Run Map Matching**.
   - ML-only scores 96.18%, HMM 98.85%.
   - The red circles mark the points the HMM changed, at the junction.
3. Choose **00000077** (3,911 points, about 1.5 s). This is the hardest trip: ML-only 76.20%, HMM 86.17%.
4. Toggle the layers: GPS, ML-only (orange dashed), HMM (green), ground truth (purple).

## Notes
- The map tiles come from OpenStreetMap, and Leaflet loads from unpkg. The browser needs internet for these; the routes are drawn even if tiles fail.
- **Verified:** localhost access, and access through Laptop 1's own LAN IP (`/health` OK on 10.5.18.34).
- **Not yet verified:** a real second-laptop test.
