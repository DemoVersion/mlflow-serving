from transformers import pipeline
import mlflow.pyfunc


class HuggingFaceWrapper(mlflow.pyfunc.PythonModel):
    """
    Class use HuggingFace Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        from transformers import pipeline

        self.model = pipeline("zero-shot-classification", model="demoversion/bert-fa-base-uncased-haddad-wikinli")

    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the FastText model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        labels = ["ورزشی",
            "سیاسی",
            "علمی",
            "فرهنگی"]
        template_str = "این یک متن {} است."
        
        return self.model(model_input['text'].tolist(), labels, hypothesis_template=template_str)