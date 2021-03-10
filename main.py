from beatTracker import beatTracker
from evaluation import evaluate
import os


inputFile = r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\chacha\Albums-Cafe_Paradiso-05.wav'
beats = beatTracker(inputFile=inputFile)



# chachacha = [r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\chacha\Albums-Cafe_Paradiso-05.wav',
#             r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\chacha\Albums-Fire-01.wav',
#              r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\chacha\Albums-Latin_Jam-01.wav',
#              r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\chacha\Albums-Latino_Latino-01.wav',
#              r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\chacha\Albums-Macumba-01.wav']
#
# waltz = [r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\waltz\Albums-Ballroom_Classics4-03.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\waltz\Albums-Ballroom_Magic-04.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\waltz\Albums-Chrisanne2-02.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\waltz\Albums-Secret_Garden-02.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\waltz\Albums-Ballroom_Magic-03.wav']
#
# jive = [r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\jive\Albums-Cafe_Paradiso-15.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\jive\Albums-Fire-12.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\jive\Albums-Latin_Jam4-05.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\jive\Albums-Macumba-15.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\jive\Albums-Pais_Tropical-15.wav']
#
# tango = [r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\tango\Albums-Ballroom_Classics4-08.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\tango\Albums-Ballroom_Magic-07.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\tango\Albums-Chrisanne3-05.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\tango\Albums-Step_By_Step-07.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\tango\Albums-StrictlyDancing_Tango-03.wav']
#
# samba = [r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\samba\Albums-Cafe_Paradiso-02.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\samba\Albums-Fire-15.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\samba\Albums-Latin_Jam3-05.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\samba\Albums-Macumba-06.wav',
#          r'C:\Users\tonyr\OneDrive\MSc\MIR\Coursework 1\evaluation\samba\Albums-Pais_Tropical-01.wav']
#
# f_mean = 0
# p_mean = 0
# counter = 0
# for filepath in chachacha:
#     print(filepath)
#     beats, x, sr, hop_size = beatTracker(filepath, return_all=True)
#     f, p = evaluate(beats, filepath)
#     f_mean += f
#     p_mean += p
#     counter += 1
#
# f_mean /= counter
# p_mean /= counter
#
# print(f"f-mean: {f_mean}")
# print(f"p-mean: {p_mean}")
