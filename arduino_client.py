import httpx


class ArduinoClient:
    """คุยกับ Arduino UNO R4 WiFi ผ่าน HTTP (endpoint ที่ฝังไว้ในสเก็ตช์ .ino)"""

    def __init__(self, ip: str, port: int = 80, timeout: float = 2.0):
        self.base_url = f"http://{ip}:{port}"
        self.timeout = timeout
        self.connected = False

    async def _get(self, path: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.text.strip()

    async def get_motion(self) -> bool:
        """เช็คค่า PIR ผ่าน /motion-status -> True ถ้ามีความเคลื่อนไหว"""
        try:
            text = await self._get("/motion-status")
            self.connected = True
            return text.strip() == "1"
        except Exception:
            self.connected = False
            return False

    async def show_helmet_ok(self) -> None:
        try:
            await self._get("/helmet-ok")
            self.connected = True
        except Exception:
            self.connected = False

    async def show_no_helmet(self) -> None:
        try:
            await self._get("/no-helmet")
            self.connected = True
        except Exception:
            self.connected = False

    async def show_idle(self) -> None:
        try:
            await self._get("/idle")
            self.connected = True
        except Exception:
            self.connected = False
