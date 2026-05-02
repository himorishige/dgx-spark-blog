"""Shared one-shot general-knowledge Q&A bank used by 05/06 scripts.

Mirrors the Phoenix handson article's "30-item general QA" set. Categories are
balanced 5-5-5-5-5-5 across geography / literature / science / history /
culture / math.
"""

GENERAL_QA: list[tuple[str, str]] = [
    # geography
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Australia?", "Canberra"),
    ("Which is the largest ocean?", "Pacific Ocean"),
    ("Which river runs through London?", "Thames"),
    ("Which mountain is the tallest above sea level?", "Mount Everest"),
    # literature
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
    ("Who wrote '1984'?", "George Orwell"),
    ("In which language was 'Don Quixote' originally written?", "Spanish"),
    ("Which Japanese author wrote 'Norwegian Wood'?", "Haruki Murakami"),
    # science
    ("What is the chemical symbol for gold?", "Au"),
    ("How many planets are in the Solar System?", "Eight"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("What is the speed of light in a vacuum (m/s)?", "299792458"),
    ("Which animal is the largest mammal?", "Blue whale"),
    # history
    ("In what year did World War II end?", "1945"),
    ("Who was the first President of the United States?", "George Washington"),
    ("In which year did the Berlin Wall fall?", "1989"),
    ("Who composed 'Symphony No. 9'?", "Ludwig van Beethoven"),
    ("In which century did the French Revolution begin?", "18th century"),
    # culture
    ("Which country gave the Statue of Liberty to the United States?", "France"),
    ("Which country invented sushi?", "Japan"),
    ("How many strings does a standard guitar have?", "Six"),
    ("Which sport uses a 'love' score?", "Tennis"),
    ("How many continents are there on Earth?", "Seven"),
    # math
    ("What is the value of pi to two decimal places?", "3.14"),
    ("How many sides does a hexagon have?", "Six"),
    ("What is 7 multiplied by 8?", "56"),
    ("Which is the smallest prime number?", "Two"),
    ("What is the square root of 144?", "12"),
]

assert len(GENERAL_QA) == 30, f"expected 30 QA, got {len(GENERAL_QA)}"
