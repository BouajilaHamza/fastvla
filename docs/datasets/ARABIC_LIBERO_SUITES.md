# Arabic LIBERO Multi-Suite Robotics Datasets

Comprehensive Arabic language annotations and multi-modal trajectory benchmarks across 30 manipulation tasks from the LIBERO benchmark:
1. **`LIBERO-10`** (`hamzabouajila/ar-libero-10-image`): Long-horizon kitchen & living room manipulation.
2. **`LIBERO-Spatial`** (`hamzabouajila/ar-libero-spatial`): Spatial relational grounding (between, next to, top drawer, inside, center).
3. **`LIBERO-Object`** (`hamzabouajila/ar-libero-object`): Fine-grained object recognition and retrieval (alphabet soup, butter, milk, bbq sauce, pudding).

---

## 1. Complete Multi-Task Taxonomy & Arabic Mappings

### A. LIBERO-Spatial Suite (10 Tasks)

| Task ID | English Ground Truth | Modern Standard Arabic (الفصحى) | Vernacular & Dialect Variations (اللهجات) |
| :---: | :--- | :--- | :--- |
| **0** | `pick up the black bowl next to the cookie box and place it on the plate` | التقط الوعاء الأسود المجاور لعلبة البسكويت وضعه على الطبق | خذ الوعاء الأسود بجانب صندوق البسكويت وحطه على الطبق / شيل الصحن الأسود اللي جنب علبة البسكوت |
| **1** | `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate` | التقط الوعاء الأسود من الدرج العلوي للخزانة الخشبية وضعه على الطبق | طلّع الوعاء الأسود من أعلى درج في الخزانة الخشبية وحطه فوق الطبق |
| **2** | `pick up the black bowl on the ramekin and place it on the plate` | التقط الوعاء الأسود الموجود فوق قالب الخَبز وضعه على الطبق | ارفع الوعاء الأسود من فوق قالب الرامكين وضعه على الطبق |
| **3** | `pick up the black bowl on the stove and place it on the plate` | التقط الوعاء الأسود الموجود على الموقد وضعه على الطبق | احمل الوعاء الأسود من فوق الغاز / البوتجاز وانقله إلى الطبق |
| **4** | `pick up the black bowl between the plate and the ramekin and place it on the plate` | التقط الوعاء الأسود الموجود بين الطبق وقالب الخَبز وضعه على الطبق | خذ الوعاء الأسود اللي بين الصحن وقالب الخَبز وضعه على الصحن |
| **5** | `pick up the black bowl on the cookie box and place it on the plate` | التقط الوعاء الأسود الموجود فوق علبة البسكويت وضعه على الطبق | شيل الوعاء الأسود من فوق كرتونة البسكويت وحطه في الطبق |
| **6** | `pick up the black bowl next to the plate and place it on the plate` | التقط الوعاء الأسود المجاور للطبق وضعه على الطبق | خذ الزبدية السوداء اللي جنب الصحن وحطها جواته |
| **7** | `pick up the black bowl next to the ramekin and place it on the plate` | التقط الوعاء الأسود المجاور لقالب الخَبز وضعه على الطبق | شيل الوعاء الأسود من جنب قالب الرامكين وانقله إلى الطبق |
| **8** | `pick up the black bowl from table center and place it on the plate` | التقط الوعاء الأسود من وسط الطاولة وضعه على الطبق | ارفع الوعاء الأسود من نص الطاولة وحطه فوق الصحن |
| **9** | `pick up the black bowl on the wooden cabinet and place it on the plate` | التقط الوعاء الأسود الموجود على الخزانة الخشبية وضعه على الطبق | انقل الوعاء الأسود من فوق الكومودينة الخشب إلى الطبق |

---

### B. LIBERO-Object Suite (10 Tasks)

| Task ID | English Ground Truth | Modern Standard Arabic (الفصحى) | Vernacular & Dialect Variations (اللهجات) |
| :---: | :--- | :--- | :--- |
| **0** | `pick up the orange juice and place it in the basket` | التقط عصير البرتقال وضعه في السلة | شيل عصير البرتقان وحطه في السبت / احمل كرتونة البرتقال للسبت |
| **1** | `pick up the ketchup and place it in the basket` | التقط الكاتشب وضعه في السلة | خذ قزازة الكاتشب وحطها في الباسكت / علبة الكاتشاب إلى السلة |
| **2** | `pick up the cream cheese and place it in the basket` | التقط الجبن الكريمي وضعه في السلة | شيل علبة الجبنة الكيري / الكريمية وحطها في السلة |
| **3** | `pick up the bbq sauce and place it in the basket` | التقط صلصة الباربكيو وضعها في السلة | خذ صوص الباربكيو وضعه داخل السلة |
| **4** | `pick up the alphabet soup and place it in the basket` | التقط حساء الحروف وضعه في السلة | شيل علبة شوربة الحروف وحطها في السلة |
| **5** | `pick up the milk and place it in the basket` | التقط الحليب وضعه في السلة | خذ علبة اللبن / الحليب وضعها في السبت |
| **6** | `pick up the salad dressing and place it in the basket` | التقط صلصة السلطة وضعها في السلة | شيل صوص السلطة وحطه في السلة |
| **7** | `pick up the butter and place it in the basket` | التقط الزبدة وضعها في السلة | خذ قالب الزبدة وضعه داخل السلة |
| **8** | `pick up the tomato sauce and place it in the basket` | التقط صلصة الطماطم وضعها في السلة | شيل معجون الطماطم / الصلصة وحطه في السبت |
| **9** | `pick up the chocolate pudding and place it in the basket` | التقط بودينغ الشوكولاتة وضعه في السلة | خذ حلوى الشوكولاتة وضعها في السلة |

---

### C. LIBERO-10 Suite (10 Tasks)

Covered in [`hamzabouajila/ar-libero-10-image`](https://huggingface.co/datasets/hamzabouajila/ar-libero-10-image):
- **Task 0**: Dual Mug Placement (`"ضع الكوب الأبيض على الطبق الأيسر والكوب الأصفر والأبيض على الطبق الأيمن"`)
- **Task 1**: Plate & Chocolate Pudding (`"ضع الكوب الأبيض على الطبق وضع حلوى الشوكولاتة على يمين الطبق"`)
- **Task 2**: Microwave Insertion (`"ضع الكوب الأصفر والأبيض في الميكروويف وأغلقه"`)
- **Task 3**: Stove & Moka Pot (`"شغّل الموقد وضع إبريق الموكا عليه"`)
- **Task 4**: Alphabet Soup & Cream Cheese (`"ضع حساء الحروف وعلبة الجبن الكريمي في السلة"`)
- **Task 5**: Alphabet Soup & Tomato Sauce (`"ضع حساء الحروف وصلصة الطماطم في السلة"`)
- **Task 6**: Dual Moka Pots (`"ضع إبريقي الموكا على الموقد"`)
- **Task 7**: Butter & Cream Cheese (`"ضع علبة الجبن الكريمي والزبدة في السلة"`)
- **Task 8**: Black Bowl Cabinet Drawer (`"ضع الوعاء الأسود في الدرج السفلي من الخزانة وأغلقه"`)
- **Task 9**: Study Caddy Book Placement (`"التقط الكتاب وضعه في الحجرة الخلفية لحامل الأدوات"`)

---

## 2. Multi-Modal Active Augmentations

During training, the collator dynamically executes:
- **`RobustRoboticsVisionAugmentor`**: $\pm 10\%$ spatial translation, color jitter (brightness, contrast, saturation $[0.8, 1.2]$), Gaussian sensor noise ($\sigma=0.02$), and $10\%$ multi-camera dropout.
- **`ArabicLanguagePerturber`**: $40\%$ dialectal synonym substitution (Egyptian, Levantine, Gulf) and $25\%$ orthographic noise (hamzat wasl/qat', taa marbuta/haa).
