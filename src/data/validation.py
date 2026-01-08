import great_expectations as ge
from great_expectations.core.expectation_configuration import ExpectationConfiguration

SUITE_NAME = "credit_data_suite"

def create_expectation_suite() -> None:
    context = ge.get_context()

    suite = context.create_expectation_suite(SUITE_NAME, overwrite_existing=True)

    expectations = [
        # Колонки существуют
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "LIMIT_BAL"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "AGE"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "default.payment.next.month"},
        ),
        # Проверка nulls
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "LIMIT_BAL"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "AGE"},
        ),
        # Диапазоны
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "AGE", "min_value": 18, "max_value": 120},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "LIMIT_BAL", "min_value": 1},
        ),
        # Таргет 0/1
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "default payment next month", "value_set": [0, 1]},
        ),
    ]

    for exp in expectations:
        suite.add_expectation(exp)

    context.save_expectation_suite(suite, SUITE_NAME)
    print(f"Saved GE suite: {SUITE_NAME}")


if __name__ == "__main__":
    create_expectation_suite()