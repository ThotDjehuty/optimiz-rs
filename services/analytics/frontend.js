import perspective from "@finos/perspective";
import viewerModule from "@finos/perspective-viewer";
import "@finos/perspective-viewer-datagrid";
import "@finos/perspective-viewer-d3fc";
import "@finos/perspective-viewer/dist/css/themes.css";
import "./style.css";

const status = document.querySelector("#status");
const source = document.querySelector("#source");
const viewer = document.querySelector("#viewer");
let socket;
let generation = 0;
let table;
let worker;

async function getJson(path) {
    const response = await fetch(path);
    if (response.status === 401) {
        location.assign("/login");
        throw new Error("Session expired");
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    return data;
}

function convert(rows) {
    return rows.map((row) => ({ ...row, event_time: new Date(row.event_time).getTime() }));
}

async function selectSource() {
    const current = ++generation;
    if (socket) socket.close();
    const old = table;
    table = await worker.table(
        { event_time: "datetime", symbol: "string", price: "float", source: "string" },
        { limit: 10000 },
    );
    const currentTable = table;
    await viewer.load(currentTable);
    if (old) await old.delete();
    await viewer.restore({ plugin: "Datagrid", columns: ["event_time", "symbol", "price", "source"], settings: true });
    if (source.value !== "live") {
        const data = await getJson(`/api/datasets/${encodeURIComponent(source.value)}`);
        if (current !== generation) return;
        await currentTable.replace(convert(data.rows));
        status.textContent = `${data.rows.length} stored rows${data.truncated ? " (truncated at limit)" : ""}`;
        return;
    }
    status.textContent = "Connecting to local bridge output…";
    socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
    let updates = Promise.resolve();
    socket.onmessage = (event) => {
        updates = updates.then(async () => {
            if (current !== generation) return;
            const message = JSON.parse(event.data);
            if (message.type === "snapshot") await currentTable.replace(convert(message.rows));
            else if (message.rows.length) await currentTable.update(convert(message.rows));
            status.textContent = `Live bridge connected · ${await currentTable.size()} buffered rows`;
        }).catch((error) => { status.textContent = error.message; });
    };
    socket.onclose = () => {
        if (current !== generation) return;
        status.textContent = "Live disconnected; reconnecting in 3 seconds…";
        setTimeout(() => {
            if (current === generation) selectSource().catch(showError);
        }, 3000);
    };
    socket.onerror = () => { status.textContent = "Live connection failed"; };
}

function showError(error) { status.textContent = error.message; }

async function main() {
    const wasm = async (name) => {
        const response = await fetch(`/static/${name}.wasm`);
        if (!response.ok) throw new Error("Local Perspective WASM assets unavailable");
        return response.arrayBuffer();
    };
    await viewerModule.init_client(await wasm("perspective-viewer"));
    perspective.init_client(await wasm("perspective-js"));
    perspective.init_server(await wasm("perspective-server"));
    worker = await perspective.worker();
    const data = await getJson("/api/datasets");
    for (const name of data.datasets) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        source.append(option);
    }
    source.onchange = () => selectSource().catch(showError);
    document.querySelector("#reload").onclick = () => selectSource().catch(showError);
    document.querySelector("#risk").onclick = async () => {
        const output = document.querySelector("#metrics");
        try {
            const query = new URLSearchParams({
                dataset_id: source.value,
                symbol: document.querySelector("#symbol").value,
                periods_per_year: document.querySelector("#periods").value,
            });
            output.textContent = JSON.stringify(await getJson(`/api/risk?${query}`), null, 2);
        } catch (error) { output.textContent = error.message; }
    };
    await selectSource();
}

main().catch(showError);
