import json


in_path: str = str(input("Enter a file path to read: "))
out_path = str(input("Enter a file path to write: "))

with open(in_path, 'r') as f_in:
  data = json.load(f_in)
  
  with open(out_path+'_separate.txt', 'w') as f_out:
    id: int = 0
    for conv in data:
      id += 1
      f_out.write(f"id={id}\n\n")
      f_out.write('seeker\n')
      f_out.write(conv['free_turker_utterance'].strip()+'\n')
      f_out.write('supporter\n')
      f_out.write(conv['guided_turker_utterance'].strip()+'\n')

      is_seeker = True
      for entry in conv['dialog']:
        if entry[0] == 0:
          f_out.write('seeker:\n')
        else:
          f_out.write('supporter:\n')
        f_out.write(entry[1].strip()+'\n')

      f_out.write('\n\n')

with open(in_path, 'r') as f_in:
  data = json.load(f_in)
  
  with open(out_path+'_merged.txt', 'w') as f_out:
    for conv in data:
      f_out.write('seeker\n')
      f_out.write(conv['free_turker_utterance'].strip()+'\n')
      f_out.write('supporter\n')
      f_out.write(conv['guided_turker_utterance'].strip()+'\n')

      is_seeker = True
      for entry in conv['dialog']:
        if entry[0] == 0:
          f_out.write('seeker:\n')
        else:
          f_out.write('supporter:\n')
        f_out.write(entry[1].strip()+'\n')