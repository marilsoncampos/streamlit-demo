"""
Application
"""
from apx_multipage import ApxMultiPage
from pages import feature_drift, label_drift

app = ApxMultiPage(image_path='res/apixio_logo.png', version_str='v1.01')
app.add_page("Feature drift", feature_drift.app)
app.add_page("Label drift", label_drift.app)

app.run()
