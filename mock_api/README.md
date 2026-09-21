# mock_api/

**Owner: M5.** The mock API that the frontend is built against until the real API exists
(PRD 18.1, 19.5 step 1). It is built from the contracts in PRD section 18 and
`docs/data-contracts.md`.

**This folder is not in the PRD repository tree (section 22.5).** It was added at setup because
the PRD says M5 builds a mock API but does not say where it lives.

Rules from the PRD:

- Every mock answer carries `"_mock": true`, so the frontend shows the MOCK DATA banner (rule H7).
- Mock data is used only while the real API is missing. No mock may remain at gate G6.
- M1 and M5 review each other's API and mock API for contract match.

Empty for now. Nothing has been implemented.
