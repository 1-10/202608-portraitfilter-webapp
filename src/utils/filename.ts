import type { OutputFormat } from "../types";

const EXTENSION_BY_FORMAT: Record<OutputFormat, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

function pad2(value: number): string {
  return value.toString().padStart(2, "0");
}

/** Formats a Date as YYYYMMDD-HHmmss in local time. */
export function formatTimestamp(date: Date = new Date()): string {
  const year = date.getFullYear();
  const month = pad2(date.getMonth() + 1);
  const day = pad2(date.getDate());
  const hours = pad2(date.getHours());
  const minutes = pad2(date.getMinutes());
  const seconds = pad2(date.getSeconds());
  return `${year}${month}${day}-${hours}${minutes}${seconds}`;
}

/** Slugifies a filter name/id for safe use inside a filename. */
function slugify(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return slug.length > 0 ? slug : "filter";
}

export function buildExportFilename(
  filterId: string,
  format: OutputFormat,
  date: Date = new Date(),
): string {
  const ext = EXTENSION_BY_FORMAT[format];
  return `portrait-${slugify(filterId)}-${formatTimestamp(date)}.${ext}`;
}
