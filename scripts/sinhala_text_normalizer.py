"""Dependency-free Sinhala text normalization shared by training/evaluation tools."""

from __future__ import annotations

import re


UNITS = ["බිංදුව", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය"]
TEENS = ["දහය", "එකොළහ", "දොළහ", "දහතුන", "දාහතර", "පහළොව", "දාසය", "දාහත", "දහඅට", "දහනවය"]
TENS = {2: "විස්ස", 3: "තිහ", 4: "හතළිහ", 5: "පනහ", 6: "හැට", 7: "හැත්තෑව", 8: "අසූව", 9: "අනූව"}
TENS_JOINED = {2: "විසි", 3: "තිස්", 4: "හතළිස්", 5: "පනස්", 6: "හැට", 7: "හැත්තෑ", 8: "අසූ", 9: "අනූ"}
HUNDREDS = {1: "එක", 2: "දෙ", 3: "තුන්", 4: "හාර", 5: "පන්", 6: "හය", 7: "හත්", 8: "අට", 9: "නව"}


def integer_to_sinhala(number: int) -> str:
    if number < 10:
        return UNITS[number]
    if number < 20:
        return TEENS[number - 10]
    if number < 100:
        tens, units = divmod(number, 10)
        return TENS[tens] if units == 0 else TENS_JOINED[tens] + UNITS[units]
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        head = "සියය" if hundreds == 1 else HUNDREDS[hundreds] + "සියය"
        if not remainder:
            return head
        head = "එකසිය" if hundreds == 1 else HUNDREDS[hundreds] + "සිය"
        return head + " " + integer_to_sinhala(remainder)
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1_000)
        if thousands == 1:
            head = "දහස" if not remainder else "එක්දහස්"
        elif thousands < 10:
            head = HUNDREDS[thousands] + ("දහස" if not remainder else "දහස්")
        else:
            head = integer_to_sinhala(thousands) + (" දහස" if not remainder else " දහස්")
        return head if not remainder else head + " " + integer_to_sinhala(remainder)
    return " ".join(UNITS[int(digit)] for digit in str(number))


def clean_input(text: str, normalize_numbers: bool = True) -> str:
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>]", "", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    if normalize_numbers:
        text = re.sub(r"\d+", lambda match: integer_to_sinhala(int(match.group())), text)
    return text

