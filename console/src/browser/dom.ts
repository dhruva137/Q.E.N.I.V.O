export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: { className?: string; text?: string; title?: string; hidden?: boolean } = {},
  ...children: Array<Node | string | null>
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (props.className) node.className = props.className;
  if (props.text != null) node.textContent = props.text;
  if (props.title) node.title = props.title;
  if (props.hidden) node.hidden = true;
  for (const child of children) {
    if (child == null) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export function clear(node: Element): void {
  node.replaceChildren();
}

export function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

export type Cell = string | { text: string; numeric?: boolean; title?: string };

export function table(headers: string[], rows: Cell[][]): HTMLTableElement {
  const tableEl = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const header of headers) {
    const th = document.createElement("th");
    th.textContent = header;
    headRow.append(th);
  }
  thead.append(headRow);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      if (typeof cell === "string") td.textContent = cell;
      else {
        td.textContent = cell.text;
        if (cell.numeric) td.className = "num";
        if (cell.title) td.title = cell.title;
      }
      tr.append(td);
    }
    tbody.append(tr);
  }
  tableEl.append(thead, tbody);
  return tableEl;
}

export function defs(pairs: Array<[string, string]>): HTMLDListElement {
  const list = document.createElement("dl");
  for (const [key, value] of pairs) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
  return list;
}
