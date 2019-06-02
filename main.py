from src import HearstPatterns

text = "Forty-four percent of patients with uveitis had one or more identifiable signs or symptoms, such as red eye, ocular pain, visual acuity, or photophobia, in order of decreasing frequency."
text2 = "A Build profile is a set of configuration values, which can be used to set or override default values of Maven build. Using a build profile, you can customize build for different environments such as Production v/s Development environments."
h = HearstPatterns()

l = h.chunk(text)

# hyps = h.find_hyponyms(text2)
# print(hyps)
