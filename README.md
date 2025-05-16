# ClipKit
<p align="center">
  <img src="https://github.com/anish9/ClipKit/blob/main/assets/clipkit_log.png" alt="ClipKit Logo" width="600"/>
</p>

### About
- ClipKit enables training customizable ```CLIP```-style contrastive models.
- The core idea behind image-text models is to combine a pre-trained vision encoder with a pre-trained text encoder.
- This project allows seamless pairing of any vision and text models with just two lines of code change.
  - ✅ Vision models: Refer to ```tf.keras.applications```
  - ✅ Text models: Use Hugging Face's ```BERT```-based models
- Community contributions are welcome to expand model support. The current implementation covers approximately 95% of common use cases.
- 💡 Example: You don’t always need a **191M parameter** (fp32) model to solve a task—often, a **50M parameter** model can deliver comparable accuracy.
   **ClipKit** encourages building compact cross-model architectures that are efficient yet powerful.

### Experiment Set-up
- ##### Set-up Virtual Env
  - ```pip install -r requirements.txt```
  - A sample dataset format is available at ```assets/demo_train_sample.csv```.
    - (The CSV should contain image paths and corresponding text captions to kickstart training.)

### Model Set-up (Customization blocks)
##### Vision Model 📷
- Import any vision model from the <a href="https://keras.io/api/applications/">Keras Applications Catalog</a>.
- In ```train.py```, you can use models like ResNet, EfficientNet, NASNet, etc.
  - ###### **Code Example**
    - ```from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0```
  - ###### **Optional Set-up**:
    - After importing edit ```get_image_model``` in ```train.py```
      - ```image_model = get_image_model(model=EfficientNetV2B0, active_layers_image_model=10)```
      - ```active_layers_image_model=10``` here last 10 layers of the image model are fine-tuned. Can be changed as needed.
      
##### Text Model 🔤
- Import any Bert family text models from <a href="https://huggingface.co/models?search=bert">Hugging Face Models Catalog</a>.
- In```train.py```
  - ###### **Code Example**
    - ```text_model_id = "huawei-noah/TinyBERT_General_4L_312D" ```
    - ```#examples: "distilbert-base-uncased","roberta", tinybert etc...```
  - ###### **Optional Set-up**:
    - After setting up ```text_model_id```, Edit ```get_text_model``` in ```train.py```. If needed turn ```trainable=False``` (Not recommended).
    - ```text_model, tokenizer = get_text_model(model_id=text_model_id, trainable=True)```

##### HyperParameters 📈
- In ```train.py``` Edit HyperParameters:
  
  - ```
        ckpt_save_dir = "my_custom_model_ckpt" #model checkpoint save directory
        model_logs_dir = "my_custom_model_logs" #tensorboard logs
        text_max_len = 12  #Maximum length of text sequence in your dataset
        batch_size = 8  
        epochs = 30
        proj_dim = 512 
        learning_rate = 5e-5
    ```  
### Launch 🚀
- After **Model Set-up**.
- Run
  ```
  python train.py
  ```
- Model logs and checkpoints will be generated once training starts successfully.

### Test 🧮
- Use ```test.ipynb``` to test your trained checkpoints.
- ##### **Example:  Zero-shot classification**: 
  - ```
    captions = [
    "the breed is shitzu",
    "the breed is norweight_elkahound",
    "the breed is Maltese", ....
    ]

    image_id = "68768d392e81a9864575a1678707565b.jpg"  # image_path
    image_vect, text_vect = get_embeddings(image_path=image_id, captions=captions)
    predictions = compute_scores(image_vector=image_vect,
                                 text_vector=text_vect,
                                 captions=captions, top_pred_count=3)
    ```   

### Philosophy
  - This project is a reliable workhorse that gets the job done. While some bug fixes will be addressed in the future, contributions and feedback are always welcome. Feel free to raise issues, suggest improvements,
    or submit pull requests to make it better.

  - The core philosophy of this repository is simple:
    **"You don’t need a chainsaw to cut a small plant."**
    In the machine learning world, this analogy reminds us that not every task requires a massive model — smaller, efficient models can often achieve the same results with less complexity and cost.
