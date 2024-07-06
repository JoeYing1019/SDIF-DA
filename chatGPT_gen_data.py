from openai import OpenAI

client = OpenAI(api_key="")
import pandas as pd
import random
from tqdm import tqdm, trange


intent = 'Greet'
example_num = 10
SYSTEM_PROMPT = "You are a helpful and intelligent dialogue utterance generation system. I will provide you the definition of a intent, you need to generate 25 non repetitive dialogue utterances containing this intent, and the maximum length (number of tokens) of utterances should under 26."

USER_PROMPT_1 = "Are you clear about your role?"

ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your utterance genration task to generate 25 non repetitive utterances and each utterance containing given intent. And i will ensure the diversity of generated utterances. Please provide me with the necessary information to get started."

GUIDELINES_PROMPT = (
    "Intent Definition:\n"
    "Greet: Express mutual kindness or recognition during the encounter (e.g., waving to someone and saying hello).\n"
    "\n"
    "Examples:\n"
    "1.Intent: Greet\n Generate Utterances:{}\n"
    "2.Intent: Greet\n Generate Utterances:{}\n"
    "\n"
    "Please help me generate:\n"
    "\n"
    "Intent: Greet\n Generate Utterances:"
)


def openai_chat_completion_response(final_prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_1},
        {"role": "assistant", "content": ASSISTANT_PROMPT_1},
        {"role": "user", "content": final_prompt}
    ])
    return response['choices'][0]['message']['content'].strip(" \n")

def get_example(file):
    examples = []
    pd_data = pd.read_csv(file, sep='\t')
    for i in range(len(pd_data)):
        text = pd_data.loc[i, "text"]
        label = pd_data.loc[i, "label"]
        if label == intent:
            examples.append(text)
    return examples




def run():
    global GUIDELINES_PROMPT, example_num, intent
    intent_examples = []
    train_examples = get_example("train.tsv")
    intent_examples.extend(train_examples)
    dev_examples = get_example("dev.tsv")
    intent_examples.extend(dev_examples)
    test_examples = get_example("test.tsv")
    intent_examples.extend(test_examples)

    f = open('{}.txt'.format(intent), 'a')

    idx = 1
    for i in tqdm(range(idx), desc='Generating'):
        examples = random.sample(intent_examples,example_num * 2)
        example_1 = '\n'.join(examples[:example_num])
        example_2 = '\n'.join(examples[example_num:])

        GUIDELINES_PROMPT = GUIDELINES_PROMPT.format(example_1, example_2)
        utts = openai_chat_completion_response(GUIDELINES_PROMPT)
        utt_list = utts.split('\n')
        for utt in utt_list:
            f.write(utt)
            f.write('\n')

run()



