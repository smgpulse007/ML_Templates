#In [1]: scene_one
# Out[1]:
# "SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: Whoa there!  [clop clop clop] \nSOLDIER #1: Halt!  Who goes there?\nARTHUR: It is I, Arthur, son of Uther Pendragon, from the castle of Camelot.  King of the Britons, defeator of the Saxons, sovereign of all England!\nSOLDIER #1: Pull the other one!\nARTHUR: I am, ...  and this is my trusty servant Patsy.  We have ridden the length and breadth of the land in search of knights who will join me in my court at Camelot.  I must speak with your lord and master.\nSOLDIER #1: What?  Ridden on a horse?\nARTHUR: Yes!\nSOLDIER #1: You're using coconuts!\nARTHUR: What?\nSOLDIER #1: You've got two empty halves of coconut and you're bangin' 'em together.\nARTHUR: So?  We have ridden since the snows of winter covered this land, through the kingdom of Mercea, through--\nSOLDIER #1: Where'd you get the coconuts?\nARTHUR: We found them.\nSOLDIER #1: Found them?  In Mercea?  The coconut's tropical!\nARTHUR: What do you mean?\nSOLDIER #1: Well, this is a temperate zone.\nARTHUR: The swallow may fly south with the sun or the house martin or the plover may seek warmer climes in winter, yet these are not strangers to our land?\nSOLDIER #1: Are you suggesting coconuts migrate?\nARTHUR: Not at all.  They could be carried.\nSOLDIER #1: What?  A swallow carrying a coconut?\nARTHUR: It could grip it by the husk!\nSOLDIER #1: It's not a question of where he grips it!  It's a simple question of weight ratios!  A five ounce bird could not carry a one pound coconut.\nARTHUR: Well, it doesn't matter.  Will you go and tell your master that Arthur from the Court of Camelot is here.\nSOLDIER #1: Listen.  In order to maintain air-speed velocity, a swallow needs to beat its wings forty-three times every second, right?\nARTHUR: Please!\nSOLDIER #1: Am I right?\nARTHUR: I'm not interested!\nSOLDIER #2: It could be carried by an African swallow!\nSOLDIER #1: Oh, yeah, an African swallow maybe, but not a European swallow.  That's my point.\nSOLDIER #2: Oh, yeah, I agree with that.\nARTHUR: Will you ask your master if he wants to join my court at Camelot?!\nSOLDIER #1: But then of course a-- African swallows are non-migratory.\nSOLDIER #2: Oh, yeah...\nSOLDIER #1: So they couldn't bring a coconut back anyway...  [clop clop clop] \nSOLDIER #2: Wait a minute!  Supposing two swallows carried it together?\nSOLDIER #1: No, they'd have to have it on a line.\nSOLDIER #2: Well, simple!  They'd just use a strand of creeper!\nSOLDIER #1: What, held under the dorsal guiding feathers?\nSOLDIER #2: Well, why not?\n"

# Import necessary modules
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)


# {"'d", 'Saxons', "'re", 'use', 'needs', 'an', 'held', 'every', 'course', 'Halt', 'Who', 'this', 'Britons', 'kingdom', 'or', 'two', 'sun', 'Listen', 'So', 'martin', 'back', 'Patsy', 'tropical', 'your', 'Whoa', 'is', 'may', 'fly', 'halves', 'weight', 'ounce', 'snows', 'covered', '#', 'then', '...', "'s", 'seek', 'A', 'in', 'by', 'right', 'these', 'Please', 'join', 'if', 'guiding', 'It', 'you', 'Supposing', 'temperate', 'European', 'migrate', '!', 'ask', 'does', 'it', 'African', 'why', 'sovereign', 'five', 'anyway', 'Where', 'Uther', 'carrying', 'Are', 'matter', 'coconut', 'through', 'from', 'second', "'m", 'court', 'You', 'yeah', 'they', 'Camelot', 'point', 'We', 'order', 'dorsal', 'our', 'Not', 'all', 'bangin', 'Pendragon', 'since', 'who', 'question', 'non-migratory', 'of', 'King', 'In', 'master', 'beat', 'zone', 'tell', 'That', 'under', 'castle', 'other', 'knights', 'KING', 'length', 'Mercea', 'get', 'carried', 'minute', '?', 'climes', 'mean', 'a', '.', 'me', 'Wait', 'son', 'Found', 'to', 'he', 'but', 'just', 'strand', 'search', 'house', 'ARTHUR', 'times', 'using', 'my', '--', 'one', 'suggesting', 'must', 'warmer', 'Court', 'there', 'I', 'air-speed', 'here', 'England', 'horse', "'em", 'found', 'ratios', 'interested', 'Arthur', 'south', 'the', 'servant', 'land', 'SOLDIER', 'be', 'forty-three', 'velocity', 'do', 'The', ',', 'simple', 'will', 'Am', 'feathers', '2', 'breadth', 'maybe', 'its', 'lord', 'got', 'Oh', 'They', 'coconuts', 'Pull', 'speak', "'ve", 'winter', 'with', 'go', ']', '[', 'grips', 'not', 'have', ':', 'carry', 'am', 'wants', 'grip', 'together', 'at', 'on', 'empty', 'them', 'bring', 'agree', 'SCENE', 'strangers', 'line', 'could', 'ridden', '1', 'swallows', 'What', 'clop', "'", 'Will', 'plover', 'But', 'where', 'defeator', 'Well', 'pound', 'that', 'No', 'goes', 'creeper', 'yet', 'Yes', "n't", 'wings', 'and', 'bird', 'maintain', 'Ridden', 'husk', 'swallow', 'are', 'wind', 'trusty'}
