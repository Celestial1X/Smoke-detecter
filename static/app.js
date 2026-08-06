document.addEventListener("DOMContentLoaded", () => {
    const violationLogs = document.getElementById("violationLogs");
    const btnClear = document.getElementById("btnClear");
    const modal = document.getElementById("imageModal");
    const imgFull = document.getElementById("imgFull");
    const caption = document.getElementById("caption");
    const closeModal = document.querySelector(".close-modal");

    // สร้างการเชื่อมต่อ WebSocket สด
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live-status`;
    let ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);

        if (response.type === "history") {
            renderHistory(response.data);
        } else if (response.type === "new_violation") {
            addNewRow(response.data);
        }
    };

    function renderHistory(records) {
        violationLogs.innerHTML = "";
        if (records.length === 0) {
            violationLogs.innerHTML = `<tr><td colspan="4" class="empty-state">ไม่พบประวัติการกระทำผิด</td></tr>`;
            return;
        }
        records.forEach(record => {
            appendRowToTable(record);
        });
    }

    function addNewRow(record) {
        // ลบข้อความไม่พบประวัติออกถ้ามีอยู่
        const emptyRow = violationLogs.querySelector(".empty-state");
        if (emptyRow) {
            violationLogs.innerHTML = "";
        }
        
        // เพิ่มแถวใหม่ไว้บนสุด
        const tr = document.createElement("tr");
        tr.className = "new-entry";
        tr.innerHTML = getRowHTML(record);
        violationLogs.insertBefore(tr, violationLogs.firstChild);

        // ใส่ Event Listener สำหรับเปิดรูปขยาย
        const img = tr.querySelector(".thumb-img");
        img.addEventListener("click", () => openModal(record.image_url, `${record.date} ${record.time} - ${record.status_text} (${record.confidence})`));
    }

    function appendRowToTable(record) {
        const tr = document.createElement("tr");
        tr.innerHTML = getRowHTML(record);
        violationLogs.appendChild(tr);

        const img = tr.querySelector(".thumb-img");
        img.addEventListener("click", () => openModal(record.image_url, `${record.date} ${record.time} - ${record.status_text} (${record.confidence})`));
    }

    function getRowHTML(record) {
        return `
            <td>${record.time}<br><small style="color:#888;">${record.date}</small></td>
            <td><span class="badge badge-danger">${record.status_text}</span></td>
            <td><strong>${record.confidence}</strong></td>
            <td>
                <img src="${record.image_url}" class="thumb-img" alt="หลักฐาน">
            </td>
        `;
    }

    // Modal Image Preview Functionality
    function openModal(src, text) {
        modal.style.display = "block";
        imgFull.src = src;
        caption.innerText = text;
    }

    closeModal.onclick = () => modal.style.display = "none";
    modal.onclick = (e) => {
        if (e.target === modal) modal.style.display = "none";
    };

    // ปุ่มล้างประวัติ
    btnClear.addEventListener("click", async () => {
        if (confirm("คุณต้องการล้างประวัติทั้งหมดใช่หรือไม่?")) {
            await fetch("/api/clear-history", { method: "DELETE" });
        }
    });
});