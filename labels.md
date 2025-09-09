# Repository Labels

This project uses GitHub Labels to organize Issues and Pull Requests.  
Below is the label dictionary for consistency.

---

## 🚀 Core Labels

### `MVP`
- **Color:** `#1d76db` (blue)
- **Description:** Minimum Viable Product tasks — required for first usable version.
- **Usage:** Assign to issues that must be finished before the first demo/release.

### `stretch`
- **Color:** `#fbca04` (yellow)
- **Description:** Nice-to-have features or enhancements that go beyond MVP scope.
- **Usage:** Assign when feature is optional or planned for future iterations.

### `bug`
- **Color:** `#d73a4a` (red)
- **Description:** A confirmed defect in the app.
- **Usage:** Assign when behavior deviates from expected results.

### `scenario`
- **Color:** `#5319e7` (purple)
- **Description:** Validation scenario for Mission Go/No-Go predictor.
- **Usage:** Assigned automatically via the Mission Scenario issue template.

### `backend`
- **Color:** `#0e8a16` (green)
- **Description:** Server-side, data ingestion, APIs, risk model code.
- **Usage:** Assign to issues that involve backend logic or services.

### `frontend`
- **Color:** `#006b75` (teal)
- **Description:** Streamlit UI, layout, charts, and visualization tasks.
- **Usage:** Assign to issues that involve user interface.

### `blocked`
- **Color:** `#b60205` (dark red)
- **Description:** Cannot progress due to external dependency (data, API, teammate).
- **Usage:** Assign when an issue is waiting on something before work can continue.

---

## 🔹 How to Use Labels
- Each issue should have at least **one label**.  
- Combine labels for clarity:  
  - Example: `MVP` + `backend`  
  - Example: `scenario` + `blocked`  
- Review labels during sprint/iteration planning to keep priorities clear.

---

## 🔹 Customization
- Labels can be edited in **Issues → Labels**.  
- Add colors that visually separate categories (e.g., `bug` red, `scenario` purple, `MVP` blue).
