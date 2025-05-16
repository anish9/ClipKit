# ClipKit
### About
      * ClipKit enables training customizable CLIP-style contrastive models.
      * The core idea behind image-text models is to combine a pre-trained vision encoder with a pre-trained text encoder.
      * This project allows seamless pairing of any vision and text models with just two lines of code change.
          ✅ Vision models: Refer to ```tf.keras.applications```
          ✅ Text models: Use Hugging Face's BERT-based models
      * Community contributions are welcome to expand model support. The current implementation covers approximately 95% of common use cases.
      * 💡 Example: You don’t always need a 191M parameter (fp32) model to solve a task—often, a 50M parameter model can deliver comparable accuracy.
         ClipKit encourages building compact cross-modal architectures that are efficient yet powerful.
