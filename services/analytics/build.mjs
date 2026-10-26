import { build } from "esbuild";
import { mkdir, copyFile } from "node:fs/promises";

await mkdir("static", { recursive: true });
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
