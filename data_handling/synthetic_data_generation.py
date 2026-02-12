class Synthetic_AAVE_Data_Generation:
      def __init__(self, path: str, data_type: str, model: str):
        self.data_type = data_type
        self.model = model
        
      def format_prompt(self, n = 250):
      prompt = ""
      if self.data_type == "interview":
        gender = random.choice(["woman", "man"])
        city = random.choices(population=["Atlanta", "DC", "Detroit", "the Lower East Side of New York City", "Princeville", "Rochester", "Valdosta"], weights=[0.05, 0.5, 0.14, 0.05, 0.11, 0.06, 0.05], k=1)[0]
        INTERVIEW_PROMPT = f"""
          You are an African American {gender} from {city}, participating in an oral sociolinguistic interview.
          Speak in a natural, conversational way, as though you are telling your life story, sharing experiences, and reflecting on everyday life.
          Produce a continuous narrative roughly the length of a 30-minute conversation (around 4,000–5,000 words).
          Include only the narrative text itself, with no headings, notes, or explanations."""
        prompt = INTERVIEW_PROMPT.format(gender=gender, city=city)
      else:
        TWEET_PROMPT = f"""
          You are given the role of a casual Twitter user. Generate {n} tweets written in African American Vernacular English (AAVE).
          Guidelines:
            - Write them in the informal, conversational style of Twitter.
            - Keep each tweet short (under 280 characters).
            - Use natural, everyday topics (music, sports, friends, emotions, funny observations, etc.).
            - Put each tweet on a newline using the newline character.
            - Include no other extraneous information about the tweet, the task or anything else. Include only the tweets.
          """
        prompt = TWEET_PROMPT.format(n=n)
      return prompt

    """
    Input: none
    Output: none
    Samples the model to create 'sociololinguistic interview' as a user from one of the cities in the human dataset - stores output in file
    If looking to extend model to one of the other models in the Together API model, you would need to receive the model tag, else must add other models API
    Must have own API key for either models to run from python terminal
    """
    def model_data(self, n = 100, data = "interview"):
      for i in range(n):
        if data == "interview":
          prompt = self.format_prompt(data = data)
        else:
          prompt = self.format_prompt(data = data, n = n)

        models = {'meta':"meta-llama/Llama-3-8b-chat-hf" , 'deepseek': 'deepseek-ai/DeepSeek-R1', 'google': 'deja_dunlap/google/gemma-2-27b-it-e3f0a0c6'}
        if self.model in ['google', 'meta']:
          client = Together(api_key=userdata.get('TOGETHER_API_KEY'))
          response = client.chat.completions.create(
              model=models[self.model],
              messages=[{"role": "user", "content": prompt}],
              temperature=1.1,
              top_p=0.95,
              presence_penalty=0.45)
        else:
          openai.api_key = userdata.get('OPENAI_API_KEY')
          client = openai.OpenAI(api_key = openai.api_key)
          response = client.chat.completions.create(
              model="gpt-4o-mini",
              store=True,
              messages=[{"role": "user", "content": prompt}])
        directory = f"{self.model}_{data}"
        if not os.path.exists(directory):
          os.makedirs(directory)
        file = directory + "/" + self.files + "_" + str(i)
        with open(file, 'w') as f:
          f.write(response.choices[0].message.content)