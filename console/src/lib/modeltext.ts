/** Same rule as server._read_model: GAMS when the text contains "Solve " and "..", otherwise MPS. */
export function guessFormat(text: string): "gams" | "mps" {
  return text.includes("Solve ") && text.includes("..") ? "gams" : "mps";
}
