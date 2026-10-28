h1. 🧪 Test-Driven Development (TDD) Strategy & Coverage Plan

|| Author | [Your Name / @Cursor] ||
|| Target Feature / Component | [Feature / API Name] ||
|| Jira Ticket(s) | [PROJ-1234] ||
|| Target Frameworks | [e.g., PyTest / Jest / JUnit / Go Test] ||
|| Target Coverage | Unit: ≥85% | Integration: ≥70% | E2E: Critical Paths ||

---

h2. 🎯 1. Overview & Acceptance Criteria
*Briefly describe the functionality being implemented and the core behavioral requirements before writing any code.*

* **Objective:** [Short explanation of what code is being added/modified]
* **Acceptance Criteria (Gherkin format):**
  * **Given** [initial context]
  * **When** [action / event]
  * **Then** [expected outcome]

---

h2. 🔴 2. Red Phase: Initial Failing Test Suites

> *Define the tests that MUST be written and fail BEFORE writing implementation code.*

h3. 2.1 Unit Tests (Happy Path & Core Logic)
|| Test ID || Test Target / Function || Input / Precondition || Expected Behavior || Test File Path ||
| UT-01 | `calculate_total()` | Valid cart items | Returns exact sum with tax | `tests/unit/test_cart.py` |
| UT-02 | `parse_payload()` | Valid JSON string | Returns structured DTO | `tests/unit/test_parser.py` |

h3. 2.2 Boundary & Edge Cases
|| Test ID || Scenario || Input Condition || Expected Handling ||
| EC-01 | Empty Input | `[]` or `null` | Throws `EmptyPayloadException` |
| EC-02 | Large Payload | 10MB JSON stream | Rejects with `413 Payload Too Large` |
| EC-03 | Concurrent Requests | 50 simultaneous calls | Handles via lock without race conditions |

h3. 2.3 Negative & Error Paths
|| Test ID || Error Scenario || Trigger Condition || Expected Exception / HTTP Code ||
| ERR-01 | Unauthorized | Missing Bearer Token | `401 Unauthorized` |
| ERR-02 | Database Timeout | DB connection drops | Retries 3x, throws `ServiceUnavailableException` |

---

h2. 🟢 3. Green Phase: Minimal Implementation Blueprint
*Outline the minimum code structure needed to transition tests from RED to GREEN.*

* **Interfaces / Signatures to Stub Out:**
{code:python}
def process_order(order_data: dict) -> OrderResult:
    # TODO: Implement minimum logic to satisfy UT-01 and UT-02
    pass
{code}

* **External Dependencies & Mocking Strategy:**
  * **Mock Service A:** Mock `PaymentGatewayClient.charge()` using `unittest.mock` / `jest.fn()`.
  * **In-Memory Storage:** Use SQLite in-memory or Redis mock container for repository testing.

---

h2. 🔵 4. Refactor Phase: Code Quality & Architecture Safety

* **Design Patterns Applied:** [e.g., Repository Pattern, Strategy Pattern]
* **Refactoring Checklist:**
  * [ ] Deduplicate setup logic using fixtures / factories.
  * [ ] Ensure all mock calls verify exact argument types and call counts.
  * [ ] Enforce async/await error isolation.
  * [ ] Clean up test side effects (teardown database state).

---

h2. 📊 5. Test Coverage Metrics & Validation Matrix

|| Component / Module || Line Coverage Target || Branch Coverage Target || Risk Level (H/M/L) ||
| `services/auth_service` | 95% | 90% | High |
| `controllers/api_v1` | 80% | 75% | Medium |
| `utils/formatter` | 100% | 100% | Low |

---

h2. 🛠️ 6. Execution Commands & CI Integration

{code:bash}
# Run unit tests in TDD watch mode
pytest --watch tests/unit/

# Run coverage reporting
pytest --cov=src --cov-report=term-missing --cov-fail-under=85
{code}