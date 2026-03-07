const TIMING_EPSILON_S = 0.03;

export function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

export function formatStamp(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const millis = Math.floor((safe - Math.floor(safe)) * 1000);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

export function formatDeltaMs(valueMs) {
  if (!Number.isFinite(valueMs)) {
    return "—";
  }
  const rounded = Math.round(valueMs);
  return `${rounded >= 0 ? "+" : ""}${rounded}ms`;
}

export function formatMs(valueMs) {
  if (!Number.isFinite(valueMs)) {
    return "—";
  }
  return `${Math.round(valueMs)}ms`;
}

export function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

export function toneFromRatio(value, goodThreshold = 0.9, watchThreshold = 0.7) {
  if (!Number.isFinite(value)) {
    return "watch";
  }
  if (value >= goodThreshold) {
    return "healthy";
  }
  if (value >= watchThreshold) {
    return "watch";
  }
  return "risk";
}

export function humanizeSegmentSourceType(value) {
  if (value === "qcom_chapter") {
    return "Quran.com chapter timestamps";
  }
  if (value === "qcom_verse") {
    return "Quran.com verse segments";
  }
  return "No direct segment supervision";
}

export function parseSourceRef(value) {
  const raw = String(value ?? "");
  if (raw.startsWith("everyayah:")) {
    const body = raw.slice("everyayah:".length);
    if (body.startsWith("http://") || body.startsWith("https://")) {
      return {
        kind: "everyayah",
        raw,
        label: `EveryAyah file • ${body}`,
      };
    }
    const parts = body.split(":");
    const fields = {};
    for (const part of parts) {
      const idx = part.indexOf("=");
      if (idx <= 0) {
        continue;
      }
      fields[part.slice(0, idx)] = part.slice(idx + 1);
    }
    const bits = [];
    if (fields.subfolder) {
      bits.push(`Folder ${fields.subfolder}`);
    }
    if (fields.surah) {
      bits.push(`Surah ${fields.surah}`);
    }
    if (fields.scope === "full_surah") {
      bits.push("Stitched full-surah reference");
    }
    return {
      kind: "everyayah",
      raw,
      label: bits.join(" • ") || body,
    };
  }

  if (raw.startsWith("qcom:")) {
    const parts = raw.split(":");
    const sourceType = parts[1] ?? "";
    const recitation = parts[2] ?? "";
    const shapeToken = parts.find((item) => item.startsWith("shape="));
    const shape = shapeToken ? shapeToken.slice("shape=".length) : "";
    const sourceLabel =
      sourceType === "qcom_chapter" ? "Quran.com chapter timestamps" : "Quran.com verse segments";
    const bits = [sourceLabel];
    if (recitation) {
      bits.push(`Recitation ${recitation}`);
    }
    if (shape) {
      bits.push(shape.replaceAll("_", " "));
    }
    return {
      kind: "qcom",
      raw,
      label: bits.join(" • "),
    };
  }

  return {
    kind: "other",
    raw,
    label: raw || "Unknown source reference",
  };
}

export function percentile(values, p) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[rank];
}

export function findSegmentIndexByTime(
  segments,
  timeS,
  { tolerance = TIMING_EPSILON_S, holdInGap = false, moveToNextInGap = false } = {},
) {
  if (!Array.isArray(segments) || segments.length === 0 || !Number.isFinite(timeS)) {
    return -1;
  }

  let left = 0;
  let right = segments.length - 1;
  let candidate = -1;
  const target = timeS + tolerance;

  while (left <= right) {
    const mid = (left + right) >> 1;
    if (segments[mid].start_s <= target) {
      candidate = mid;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  if (candidate < 0) {
    return -1;
  }

  const current = segments[candidate];
  if (timeS >= current.start_s - tolerance && timeS <= current.end_s + tolerance) {
    return candidate;
  }

  const next = segments[candidate + 1];
  if (timeS > current.end_s && (!next || timeS < next.start_s)) {
    if (moveToNextInGap) {
      return next ? candidate + 1 : candidate;
    }
    if (holdInGap) {
      return candidate;
    }
  }

  return -1;
}
