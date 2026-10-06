export interface LibraryParameter {
  name: string;
  default: string;
}

export interface LibraryModel {
  name: string;
  sector: string;
  problemClass: string;
  formulation: string;
  summary: string;
  detail: string;
  parameters: LibraryParameter[];
  command: string;
}
