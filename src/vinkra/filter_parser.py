# TODO: Use the third party regex ibstead
import re

from vinkra.exceptions import FilterError


class FilterToSql:
    """
    Guide to filter expressions

    Fields:
        - Names: alphanumeric/Unicode, e.g., category, price, last_login
        - Values:
             - strings: 'text' or "text"
             - numbers: 42 or 3.14
             - boolean values: True or False

    Operators: `==`, `!=`, `>`, `<`, `>=`, `<=`

    Examples:
        category == 'science'
        price >= 10
        value == 50
        in_stock == True
    """

    r_space = re.compile(r"\s*")
    r_ident = re.compile(r"[^\W\d]\w*")
    r_ops = re.compile(r"(?:>=|<=|==|!=|>|<)")
    r_string = re.compile(r"(['\"])(?:\\\1|.)*?\1")
    r_num = re.compile(
        # Float (e.g., 123.45, -0.5, .25, 1.2e3)
        r"(?:"
        r"[-+]?(?:"
        r"(?:\d*\.\d+|\d+\.\d*)(?:[eE][-+]?\d+)?|"  # Decimals with optional exponent
        r"\d+[eE][-+]?\d+"  # Integers with mandatory exponent (e.g. 1e3)
        r")"
        r"(?![\w\.])"  # Prevents letters, underscore, or dot right after
        r")|"
        # Int (e.g., 123, -42, +7)
        r"(?:[-+]?\d+(?![\w\.]))"
    )
    r_bool_val = re.compile(r"\b(?:True|False)\b")

    START = [
        ((r_ident,), "field name (e.g., category, content)"),
        ((r_ops,), "comparison operator (==, !=, >=, <=, >, <)"),
        (
            (r_string, r_num, r_bool_val),
            "quoted string, number or a titled boolean value",
        ),
    ]

    def translate(self, filters: list[str]) -> tuple[str, list]:
        """
        Convert a list of filter strings into a safe SQLite query.

        Args:
            filters (list[str]): List of filter strings.

        Returns:
            tuple[str, list]: A tuple containing:
                - query (str): The generated SQLite condition clause.
                - params (list): List of parameters to safely bind to the query.
        """
        all_conditions = []
        query_params = []

        for idx, line in enumerate(filters):
            res = self._parse_expression(line)

            if not res["success"]:
                raise FilterError(
                    f"error at index {idx}, col {res['col']}, found: {res['found']}, expecting: {res['expect']}"
                )

            sequence = res["sequence"]

            field = sequence[0]
            if field == "content":
                field = "content_fts5"
            else:
                field = f"metadata ->> '{field}'"

            operator = (
                "=" if sequence[1] == "==" else sequence[1]
            )  # Normalize to sql syntax
            all_conditions.append(f"{field} {operator} ?")

            literal = sequence[2]
            if literal in {"True", "False"}:
                literal = 1 if literal == "True" else 0
            else:
                literal = self._cast_value(sequence[2])
            query_params.append(literal)

        if not all_conditions:
            return "", []

        nl = "\n"
        final_expr = f"{nl.join(f'({cond})' for cond in all_conditions)}"
        return final_expr, query_params

    def _parse_expression(self, line: str) -> dict:
        """Parse a single filter expression into a sequence of tokens.
        Args:
            line (str): Filter expression string.
        Returns:
            dict: Success dict with 'sequence' key, or error dict with 'expect' key.
        """
        curr_col = 0
        curr_substring = line
        sequence = []
        for pat, expect in self.START:
            has_match = False
            for alt in pat:
                # Handle whitespace
                num_spaces = len(curr_substring) - len(curr_substring.lstrip())
                curr_col += num_spaces
                curr_substring = curr_substring[num_spaces:]

                if not curr_substring:
                    return {
                        "success": False,
                        "expect": expect,
                        "found": "end of input",
                        "col": curr_col,
                    }
                m = alt.match(curr_substring)
                if not m:
                    has_match = False
                    continue

                has_match = True
                sequence.append(m.group())
                match_len = m.span()[1]
                curr_substring = curr_substring[match_len:]
                curr_col += match_len
                break

            if not has_match:
                found = curr_substring.split()[0]
                return {
                    "success": False,
                    "expect": expect,
                    "found": found,
                    "col": curr_col,
                }

        if curr_substring:
            return {
                "success": False,
                "expect": "end of statement",
                "found": curr_substring.strip(),
                "col": curr_col,
            }

        return {"success": True, "sequence": sequence}

    def _cast_value(self, val: str):
        """Cast a value to a string or number else keep as is."""
        val = val.strip("'\"")
        try:
            if any(char in val for char in ".[eE]"):
                return float(val)
            return int(val)
        except ValueError:
            return val


if __name__ == "__main__":  # pragma: no cover
    trans = FilterToSql()

    filters = [
        "category == 'science'",
        "price >= 10",
        "rating > 4.5",
        "in_stock == True",
    ]

    print("Demonstrating FilterToSql:\n")
    for f in filters:
        query, params = trans.translate([f])
        print(f"Filter: {f}")
        print(f"SQL cond:    {query}")
        print(f"Params: {params}\n")
