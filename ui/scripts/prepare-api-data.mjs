import { cp, rm, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const thisFile = fileURLToPath(import.meta.url);
const thisDir = path.dirname(thisFile);
const repoRoot = path.resolve(thisDir, "..", "..");

const sourceApiRoot = path.join(repoRoot, "data", "api");
const sourceRecitersIndex = path.join(repoRoot, "data", "reciters.json");
const targetDataRoot = path.join(repoRoot, "ui", "public", "data");

async function ensureExists(target, label) {
  try {
    await stat(target);
  } catch {
    throw new Error(
      `${label} not found: ${target}. Run \`uv run qad build-api --export-only\` first.`,
    );
  }
}

async function main() {
  await ensureExists(sourceApiRoot, "API data directory");
  await ensureExists(sourceRecitersIndex, "reciters index file");

  await rm(targetDataRoot, { recursive: true, force: true });
  await mkdir(targetDataRoot, { recursive: true });

  await cp(sourceApiRoot, targetDataRoot, { recursive: true });
  await cp(sourceRecitersIndex, path.join(targetDataRoot, "reciters.json"));

  console.log("[prepare-api-data] synced data/api + data/reciters.json to ui/public/data");
}

main().catch((error) => {
  console.error(`[prepare-api-data] ${error.message}`);
  process.exit(1);
});
