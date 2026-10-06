# Ground reality: how an Indian refinery would actually run this

This page records what we could establish from public sources about how MRPL and its peers run
planning software, and what each fact means for QENIVO's design. Each line is marked **Found**
(a public source says so) or **Assumed** (the normal practice, to confirm with MRPL's planning and
IT teams). Snapshot: 30 Sep 2026.

## 1. What they use today

| Fact | Status | Source |
|---|---|---|
| Indian refineries plan with LP planning systems. BPCL Kochi uses Aspen PIMS to maximise net realisation and GRM; IOCL built refinery LP models for five refineries (Panipat, Koyali, Barauni, Mathura, Haldia) on Honeywell's RPMS. The other common tool is Haverly's GRTMPS | Found | IOCL "Manthan" and BPCL planning references, summarised in public refinery-planning literature |
| These tools are successive-LP systems: they generate an LP matrix, hand it to a commercial solver (CPLEX, Xpress and similar are the usual back ends), then update the nonlinear pooling terms and solve again (distributive recursion) | Found | AspenTech product pages for Aspen PIMS and PIMS-AO; refinery LP modelling literature |
| MRPL runs SAP for production and sales, and has moved its ERP to SAP S/4HANA on a cloud platform (target completion Nov 2025) | Found | MRPL e-Governance page; MRPL annual report |
| MRPL's SAP data centre and disaster-recovery centre hold ISO 27001:2013 certification | Found | MRPL e-Governance page |
| MRPL's plant control uses Honeywell systems (MasterLogic PLCs; UniSim Design for process simulation) | Found | Honeywell MRPL case study |
| Planners work on Windows desktops, with Excel as the everyday front end | Assumed | Normal practice for PIMS/RPMS users (both are Windows applications with Excel data tables) |
| MRPL's webmail is served from `webmail.mrpl.co.in/iwaredir.nsf`, the redirect page of HCL (formerly IBM Lotus) Domino iNotes, so mail and its directory run on Domino | Found | Public MRPL login pages |
| SAP is reached through the "mPower" portal (`mpower.mrpl.co.in`); in-house web applications run on `mrplapps.mrpl.co.in` (Java Server Faces, `.xhtml`) with their own username/password log-in; the intranet is called MRPLNET | Found | Public MRPL login pages |
| There is one company directory behind those log-ins (Domino LDAP, Microsoft Active Directory, or both) | Assumed | Confirm with MRPL IT; the design works with either (next section) |
| Planning servers have CPUs; datacentre GPUs are not standard in refinery IT | Assumed | Confirm; this is why every QENIVO engine runs on the CPU and the GPU is an accelerator, never a requirement |

## 2. The rules they operate under

| Rule | Status | Source |
|---|---|---|
| OT (control) networks must be hard-isolated from internet-facing IT systems | Found | CEA cyber-security guidelines 2021 for the power sector; the same practice in refinery OT (IEC 62443 segmentation) |
| CERT-In's April 2022 directions require logs to be kept for 180 days | Found | CERT-In directions under section 70B, IT Act |
| Critical infrastructure falls under NCIIPC protection; security audits of software are expected | Found | NCIIPC / CERT-In framework summaries |

## 3. What this means for the design, and where each point is met

| Ground reality | Design decision | Where in the code |
|---|---|---|
| Existing planning tools produce an LP/MPS matrix for an external solver | Read and write free MPS (with OBJSENSE, RANGES, integer markers and QUADOBJ), so a site can pass the same matrix it sends CPLEX/Xpress today. Distributive recursion is implemented natively too | `io/mps.py`, `workload/recursion.py` |
| GAMS-style models in the literature | GAMS reader for the open refinery benchmark | `io/gams.py` |
| Planners live in Excel | What-if case tables (the PIMS CASE idea) read from and written to `.xlsx` or CSV with no extra software | `io/sheets.py`, `qenivo cases model.mps cases.xlsx --out results.xlsx` |
| Air-gapped planning network | Offline installation bundle (every wheel, SHA-256 sums, an SPDX SBOM); no telemetry; no outbound connection anywhere in the code | `packaging/build_offline_bundle.py` |
| Windows desktops | Pure Python + NumPy/SciPy, spawn-based multi-core that behaves the same on Windows; a `.bat` launcher | `workload/parallel.py`, `packaging/windows/` |
| Corporate sign-in (Domino and/or Active Directory) | No new passwords and no directory code in the solver: the site's reverse proxy authenticates (IIS with Windows Authentication for AD; Apache `mod_authnz_ldap` against the Domino or AD LDAP directory; Kerberos) and passes the user name in a header trusted only from that proxy; otherwise a bearer token for a shared server | `security.py`, `packaging/iis/web.config`, `docs/DEPLOYMENT.md` |
| CERT-In 180-day logs | Append-only JSONL audit of every request (user, client, model SHA-256, engine, verdict, objective, time), monthly files, retention purge at 180 days by default | `security.py` |
| Security review of new software | Provenance guard (no solver library can load in the solve path), SBOM, answers that anyone can re-check with the independent verifier | `provenance.py`, `certify/verify.py` |
| SAP holds prices, demands and stock | File-based exchange first (SAP exports CSV; the case table reads it). A direct SAP connection is a site integration job, not solver work | `io/sheets.py` |
| GPUs uncertain | CPU is the default path; the GPU is used only where it measurably wins (batched what-ifs, very large LPs), chosen by the router with the reason recorded | `workload/router.py` |

## 4. Open questions for MRPL (ask at the first review)

1. Which planning system runs today (PIMS, RPMS, GRTMPS), and can it export its matrix as MPS?
2. Which solver does it call, and how long does a monthly plan and a case stack take today?
3. Is there an approved Linux or Windows server for planning, and does it have a GPU?
4. Which directory do internal web tools authenticate against (Domino LDAP, Active Directory, ADFS)?
5. Where must audit logs go (local files, a SIEM collector), and for how long?
