import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const script = readFileSync(new URL("./frontend.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "")
    .replace("main().catch(showError);", "");

function deferred() {
    let resolve;
    const promise = new Promise((done) => { resolve = done; });
    return { promise, resolve };
}

test("overlapping selections dispose stale tables without deleting the active table twice", async () => {
    const pendingTable = deferred();
    const tables = [];
    const loaded = [];
    let count = 0;
    const source = { value: "first.csv" };
    const status = { textContent: "" };
    const worker = {
        async table() {
            const current = {
                id: ++count,
                deleted: 0,
                async delete() {
                    this.deleted += 1;
                    assert.equal(this.deleted, 1, "a table must never be deleted twice");
                },
                async replace() {},
            };
            tables.push(current);
            if (count === 1) await pendingTable.promise;
            return current;
        },
    };
    const viewer = {
        async load(table) { loaded.push(table.id); },
        async restore() {},
    };
    const context = vm.createContext({
        document: { querySelector: (selector) => ({ "#source": source, "#status": status, "#viewer": viewer }[selector]) },
        fetch: async () => ({ ok: true, status: 200, json: async () => ({ rows: [], truncated: false }) }),
        stubWorker: worker,
    });
    vm.runInContext(script + "\nworker = stubWorker;", context);
    const first = vm.runInContext("selectSource()", context);
    await new Promise((done) => setImmediate(done));
    source.value = "second.csv";
    const second = vm.runInContext("selectSource()", context);
    source.value = "third.csv";
    const third = vm.runInContext("selectSource()", context);
    pendingTable.resolve();
    await Promise.all([first, second, third]);
    assert.deepEqual(loaded, [2]);
    assert.equal(tables[0].deleted, 1);
    assert.equal(tables[1].deleted, 0);
    assert.equal(status.textContent, "0 stored rows");
    await vm.runInContext("selectSource()", context);
    assert.equal(tables[1].deleted, 1);
    assert.equal(tables[2].deleted, 0);
});
