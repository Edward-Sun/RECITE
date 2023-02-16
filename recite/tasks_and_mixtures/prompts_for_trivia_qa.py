from tasks_and_mixtures.prompts import RECITE_PROMPT

TRIVIAQA_FIXED_RECITE_PROMPT = f"""Question: Triggered by Rosa Parks' refusal to give up her seat, the public transportation system in what US city was devastated by a year long boycott of their busses?

{RECITE_PROMPT}

Answer: Rosa Parks --- Short description --- Paragraph #2
On December 1, 1955, in Montgomery, Alabama, Parks rejected bus driver James F. Blake's order to vacate a row of four seats in the "colored" section in favor of a White passenger, once the "White" section was filled.


Question: What was Beethoven's last symphony?

{RECITE_PROMPT}

Answer: Ludwig van Beethoven --- Life and career --- 1823-1826: The final years --- Paragraph #1
The year 1823 saw the completion of three notable works, all of which had occupied Beethoven for some years, namely the Missa solemnis, the Ninth Symphony and the Diabelli Variations.


Question: In which 1972 John Boorman film is a leading character, played by Ned Beatty, raped by a 'Hillbilly'?

{RECITE_PROMPT}

Answer: Ned Beatty --- Career --- 1970s --- Paragraph #1
In 1972, Beatty made his film debut as Bobby Trippe in Deliverance, starring Jon Voight and Burt Reynolds, and set in northern Georgia. Beatty's character is forced to strip at gunpoint by two mountain men who humiliate and rape him, a scene so shocking that it is still referred to as a screen milestone.


Question: Which bridge crossing The River Thames did Queen Elizabeth II open on 17th March 1973?

{RECITE_PROMPT}

Answer: March 1973 --- March 17, 1973 (Saturday) -- Paragraph #1
Queen Elizabeth II of the United Kingdom opens the new London Bridge.


Question: "The song ""My Kind Of Town"", written by Sammy Cahn and Jimmy Van Heusen in 1964, was about which city?"

{RECITE_PROMPT}

Answer: My Kind of Town --- Short description --- Paragraph #3
"My Kind of Town" made a minor appearance on the U.S. pop charts, reaching #110 in 1964. It was the second of two charting songs about Chicago recorded by Sinatra. The other was "Chicago (That Toddlin' Town)" from 1957, which reached U.S. #84."""

TRIVIAQA_FIXED_DIRECT_PROMPT = """Q: Triggered by Rosa Parks' refusal to give up her seat, the public transportation system in what US city was devastated by a year long boycott of their busses?

A: Montgomery, Al


Q: What was Beethoven's last symphony?

A: 9th


Q: In which 1972 John Boorman film is a leading character, played by Ned Beatty, raped by a 'Hillbilly'?

A: Deliverance


Q: Which bridge crossing The River Thames did Queen Elizabeth II open on 17th March 1973?

A: London Bridge


Q: "The song ""My Kind Of Town"", written by Sammy Cahn and Jimmy Van Heusen in 1964, was about which city?"

A: Chicago"""
