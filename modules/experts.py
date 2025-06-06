def get_buffett_prompt(fundamentals):
    return f"""
    As Warren Buffett, analyze the following stock fundamentals and explain whether this company has a durable competitive advantage and is a good value investment:
    {fundamentals}
    """

def get_lynch_prompt(fundamentals):
    return f"""
    As Peter Lynch, analyze the following stock fundamentals and explain whether this company fits the profile of a 'tenbagger' growth stock:
    {fundamentals}
    """

def get_dalio_prompt(fundamentals):
    return f"""
    As Ray Dalio, analyze the following stock fundamentals and discuss the macroeconomic and risk factors that could impact this company:
    {fundamentals}
    """

def generate_expert_analysis(fundamentals, llm, expert='buffett'):
    if expert == 'buffett':
        prompt = get_buffett_prompt(fundamentals)
    elif expert == 'lynch':
        prompt = get_lynch_prompt(fundamentals)
    elif expert == 'dalio':
        prompt = get_dalio_prompt(fundamentals)
    else:
        raise ValueError('Unknown expert')
    return llm(prompt) 