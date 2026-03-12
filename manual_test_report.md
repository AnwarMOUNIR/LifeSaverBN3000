# Manual Frontend Testing Report

**Project:** LifeSaverBN3000 - Obesity Risk Prediction
**Date:** March 12, 2026 - 12:03
**Tester:** Meryem Soussi
**Role:** QA & Testing Engineer

##  EXECUTIVE SUMMARY
**All tests PASSED successfully!** The frontend application is working perfectly and ready for deployment.

---

##  TEST ENVIRONMENT
- **Application:** Streamlit Frontend
- **Browser:** [e.g., Chrome/Firefox/Edge]
- **OS:** Windows 11
- **Local URL:** https://lifesaverbn3000-a8zbdfb3hpmhssuql2z5sx.streamlit.app/#obesity-risk-prediction

---

##  DETAILED TEST RESULTS

### 1. Application Launch
| Test | Result | Notes |
|------|--------|-------|
| App starts without errors | PASS | `streamlit run app/app.py` works |
| Browser opens automatically | PASS | Local URL loads correctly |
| Page loads in < 5 seconds | PASS | Good performance |

### 2. User Interface
| Test | Result | Notes |
|------|--------|-------|
| Title displays correctly | PASS | "Obesity Risk Prediction" visible |
| All input fields present | PASS | Age, Height, Weight, etc. |
| Labels are clear | PASS | Easy to understand |
| Layout is professional | PASS | Clean and organized |

### 3. Input Forms
| Test | Result | Notes |
|------|--------|-------|
| Age field (numeric) | PASS | Accepts 25, 35, 45 etc. |
| Height field (numeric) | PASS | Accepts 1.75, 1.80 etc. |
| Weight field (numeric) | PASS | Accepts 70.5, 85.3 etc. |
| Gender dropdown | PASS | Male/Female selectable |
| Family history radio | PASS | Yes/No works |
| All categorical fields | PASS | FAVC, SMOKE, etc. all work |

### 4. Form Validation
| Test | Result | Notes |
|------|--------|-------|
| Empty fields | PASS | Shows friendly error message |
| Negative numbers | PASS | Rejected with error |
| Text in number fields | PASS | Rejected with error |
| Out of range values | PASS | Handles gracefully |

### 5. Prediction Functionality
| Test | Result | Notes |
|------|--------|-------|
| "Predict" button works | PASS | Click triggers prediction |
| Loading indicator shows | PASS | User knows it's working |
| Result displays clearly | PASS | Obesity level shown |
| Result is logical | DID NOT PASS | Healthy inputs → Normal Weight |

### 6. SHAP Explanations
| Test | Result | Notes |
|------|--------|-------|
| SHAP visualizations load | PASS | Graphs appear below prediction |
| Summary plot displays | PASS | Feature importance visible |
| Individual explanation | PASS | Single patient explanation works |
| Plots are readable | PASS | Clear and well-formatted |

### 7. Responsive Design
| Test | Result | Notes |
|------|--------|-------|
| Desktop view (>1200px) | PASS | Full layout works |
| Tablet view (768px) | PASS | Adapts well |
| Mobile view (375px) | PASS | Still usable |

### 8. Edge Cases Tested
| Test | Result | Notes |
|------|--------|-------|
| Very young age (10) | PASS | Handled appropriately |
| Very old age (80) | PASS | Handled appropriately |
| Extremely low weight (30kg) | PASS | Handled appropriately |
| Extremely high weight (200kg) | PASS | Handled appropriately |

---

---

## CONCLUSION

| Aspect | Status |
|--------|--------|
| **All tests** | ALL PASSED BUT 1 |
| **Bugs found** | 1 |
| **Blocking issues** | None |
| **Ready for deployment** | NO |

**Final Verdict:** The frontend application doesn't detect underweight cases sometimes

---

