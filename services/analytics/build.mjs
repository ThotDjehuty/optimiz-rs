import { build } from "esbuild";
import { mkdir, copyFile, readdir } from "node:fs/promises";

await mkdir("static", { recursive: true });
await mkdir("static/licenses", { recursive: true });
for (const pkg of await readdir("node_modules/@finos")) {
    const directory = `node_modules/@finos/${pkg}`;
    await copyFile(`${directory}/LICENSE.md`, `static/licenses/${pkg}.txt`);
    for (const file of await readdir(directory)) {
        if (/^(NOTICE|COPYRIGHT)(\..*)?$/i.test(file)) {
            await copyFile(`${directory}/${file}`, `static/licenses/${pkg}-${file}`);
        }
    }
}
for (const [pkg, file] of [
    ["perspective", "perspective-js.wasm"],
    ["perspective", "perspective-server.wasm"],
    ["perspective-viewer", "perspective-viewer.wasm"],
]) {
    await copyFile(`node_modules/@finos/${pkg}/dist/wasm/${file}`, `static/${file}`);
}
await build({
    entryPoints: ["frontend.js"],
    bundle: true,
    format: "esm",
    platform: "browser",
    target: "es2022",
    outfile: "static/app.js",
    minify: true,
    legalComments: "linked",
});
