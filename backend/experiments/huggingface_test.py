from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Use the correct model class for T5-based models
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create the text generation pipeline
# For seq2seq models, we use "text2text-generation" instead of "text-generation"
pl = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

question = "What are the symptoms of diabetes?"
context = "Diabetes is a metabolic disease that causes high blood sugar. The symptoms include increased thirst, frequent urination, and unexplained weight loss."

# Format the prompt
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
answer = pl(prompt, max_length=200, do_sample=True, temperature=0.7)

print(answer[0]['generated_text'])