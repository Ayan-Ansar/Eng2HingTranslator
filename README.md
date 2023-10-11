# English to Hinglish Machine Translation using T5-Small

This project utilizes the Hugging Face Transformers library and the pretrained model "[t5-small](https://huggingface.co/t5-small)" to perform English to Hinglish machine translation. The model is fine-tuned for the specific use case and trained on the [findnitai/english-to-hinglish dataset](https://huggingface.co/datasets/findnitai/english-to-hinglish).

## Instructions

To train the model for English to Hinglish translation, follow these steps:

1. Open the `model_train.ipynb` notebook in Google Colab.
2. Execute the cells in the notebook to train the model using the provided dataset.

For performing inference (translation), follow these steps:

1. Open the `inference.ipynb` notebook in Google Colab.
2. Execute the cells in the notebook to use the trained model for translating English text to Hinglish.
3. ```python
    def translate(input_text):
        tokenized = tokenizer([input_text], return_tensors='np')
        out = model.generate(**tokenized, max_length=128)
        with tokenizer.as_target_tokenizer():
            result = tokenizer.decode(out[0], skip_special_tokens=True)
        print('Translated:', result)
        result = result.lower().split(' ')
        ans = []
        for i in result:
            if i in common_hindi_eng_words:
                i = translate_to_hindi(i)
            if i not in words.words():
                i = translate_to_hindi(i)
            ans.append(i)
    
        return ' '.join(ans)
    ```
    You can use this function to perform English to Hinglish translation using the trained model.

## OUTPUT
[output_image]()



    
   