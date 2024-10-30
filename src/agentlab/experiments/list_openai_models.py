import pandas as pd
from openai import OpenAI

if __name__ == "__main__":
    models = OpenAI(base_url="https://api.shubiaobiao.cn/v1/").models.list()
    df = pd.DataFrame([dict(model) for model in models.data])

    # Filter GPT models
    df = df[df["id"].str.contains("gpt")]

    # Convert Unix timestamps to dates (YYYY-MM-DD) and remove time
    df["created"] = pd.to_datetime(df["created"], unit="s").dt.date
    df.sort_values(by="created", inplace=True)
    # Print all entries
    print(df)
