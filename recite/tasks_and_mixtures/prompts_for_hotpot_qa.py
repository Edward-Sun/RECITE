from tasks_and_mixtures.prompts import HOTPOT_RECITE_PROMPT

HOTPOTQA_FIXED_DIRECT_PROMPT = """Q: Which magazine was started first Arthur’s Magazine or First for Women?

A: The answer is Arthur’s Magazine.


Q: The Oberoi family is part of a hotel company that has a head office in what city?

A: The answer is Delhi.


Q: What nationality was James Henry Miller’s wife?

A: The answer is American.


Q: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?

A: The answer is 2006."""

HOTPOTQA_FIXED_COT_PROMPT = """Q: Which magazine was started first Arthur’s Magazine or First for Women?

A: Arthur’s Magazine started in 1844. First for Women started in 1989. So Arthur’s Magazine was started first. The answer is Arthur’s Magazine.


Q: The Oberoi family is part of a hotel company that has a head office in what city?

A: The Oberoi family is part of the hotel company called The Oberoi Group. The Oberoi Group has its head office in Delhi. The answer is Delhi.


Q: What nationality was James Henry Miller’s wife?

A: James Henry Miller’s wife is June Miller. June Miller is an American. The answer is American.


Q: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?

A: "House of Anubis" is based on the Dutch–Belgian television series Het Huis Anubis. Het Huis Anubis is firstaired in September 2006. The answer is 2006."""

HOTPOTQA_FIXED_RECITE_PROMPT = f"""Question: Which magazine was started first Arthur’s Magazine or First for Women?

{HOTPOT_RECITE_PROMPT}

Answer 1: Arthur magazine was a bi-monthly periodical that was founded in October 2002, by publisher Laris Kreslins and editor Jay Babcock.

Answer 2: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989.


Question: The Oberoi family is part of a hotel company that has a head office in what city?

{HOTPOT_RECITE_PROMPT}

Answer 1: P.R.S. Oberoi is the current chairman of The Oberoi Group.

Answer 2: The Oberoi Group is an award-winning luxury hotel group with its head office in New Delhi, India.


Question: What nationality was James Henry Miller’s wife?

{HOTPOT_RECITE_PROMPT}

Answer 1: In 1967, Miller married his fifth wife, Japanese born singer Hoki Tokuda.

Answer 2: Hoki Tokuda is an Japanese actress, known for Blind Woman's Curse (1970), The Abalone Girls (1965) and Nippon Paradise (1964). 


Question: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?

{HOTPOT_RECITE_PROMPT}

Answer 1: House of Anubis is a mystery television series developed for Nickelodeon based on the Dutch–Belgian television series Het Huis Anubis.

Answer 2: Het Huis Anubis (English: The House of Anubis) is a Dutch-Belgian children's television mystery drama. It first aired in September 2006."""
