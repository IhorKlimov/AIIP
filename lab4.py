import dlib
import cmake
import cv2
from skimage import io
from scipy.spatial import distance
import requests
from bs4 import BeautifulSoup

Dictionary_Kafedra_ = {}
ist_data = {
    "Ролік Олександр Іванович": "https://ist.kpi.ua/wp-content/uploads/2021/09/rolik.jpg",
    "Дорошенко Анатолій Юхимович": "https://ist.kpi.ua/wp-content/uploads/2021/09/doroshenko.jpg",
    "Жураковський Богдан Юрійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/zhurakovsky.jpg",
    "Корнага Ярослав Ігорович": "https://ist.kpi.ua/wp-content/uploads/2022/11/kornaga.png",
    "Корнієнко Богдан Ярославович": "https://ist.kpi.ua/wp-content/uploads/2021/09/korniienko.jpg",
    "Онищенко Вікторія Валеріївна": "https://ist.kpi.ua/wp-content/uploads/2021/09/onyshchenko.jpg",
    "Стенін Олександр Африканович": "https://ist.kpi.ua/wp-content/uploads/2021/09/stenin.jpg",
    "Теленик Сергій Федорович": "https://ist.kpi.ua/wp-content/uploads/2021/09/telenyk.jpg",
    "Шемаєв Володимир Миколайович": "https://ist.kpi.ua/wp-content/uploads/2021/09/shemaiev.jpg",
    "Амонс Олександр Анатолійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/amons.jpg",
    "Батрак Євгеній Олександрович": "https://ist.kpi.ua/wp-content/uploads/2022/11/batrak.jpg",
    "Богданова Наталія Володимирівна": "https://ist.kpi.ua/wp-content/uploads/2021/12/bogdanova2.jpg",
    "Бойко Олександра Володимирівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/boyko.jpg",
    "Букасов Максим Михайлович": "https://ist.kpi.ua/wp-content/uploads/2021/09/bukasov.jpg",
    "Гавриленко Олена Валеріївна": "https://ist.kpi.ua/wp-content/uploads/2022/11/gavrilenko1.png",
    "Дорогий Ярослав Юрійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/dorohyi.jpg",
    "Жданова Олена Григорівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/zhdanova.jpg",
    "Жереб Костянтин Анатолійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/zhereb.jpg",
    "Жураковська Оксана Сергіївна": "https://ist.kpi.ua/wp-content/uploads/2021/12/zhurakovska.jpg",
    "Завгородній Валерій Вікторович": "https://ist.kpi.ua/wp-content/uploads/2021/12/zavgorodniiv.jpg",
    "Завгородня Ганна Анатоліївна": "https://ist.kpi.ua/wp-content/uploads/2021/12/zavgorodnyaga.jpg",
    "Катін Павло Юрійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/katin.jpg",
    "Ковальов Микола Олександрович": "https://ist.kpi.ua/wp-content/uploads/2022/11/kovalov.jpg",
    "Коган Алла Вікторівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/kogan.jpg",
    "Кравець Петро Іванович": "https://ist.kpi.ua/wp-content/uploads/2021/09/kravets.jpg",
    "Крилов Євген Володимирович": "https://ist.kpi.ua/wp-content/uploads/2021/09/krylov.jpg",
    "Махно Таісія Олександрівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/makhno.jpg",
    "Мелкумян Катерина Юріївна": "https://ist.kpi.ua/wp-content/uploads/2021/09/melkumyan.jpg",
    "Новацький Анатолій Олександрович": "https://ist.kpi.ua/wp-content/uploads/2021/09/novatskyi.jpg",
    "Олійник Володимир Валентинович": "https://ist.kpi.ua/wp-content/uploads/2021/09/oleynik.jpg",
    "Остапченко Костянтин Борисович": "https://ist.kpi.ua/wp-content/uploads/2021/09/ostapchenko.jpg",
    "Пасько Віктор Петрович": "https://ist.kpi.ua/wp-content/uploads/2021/09/pasko.jpg",
    "Писаренко Андрій Володимирович": "https://ist.kpi.ua/wp-content/uploads/2021/09/pysarenko.jpg",
    "Поліщук Михайло Миколайович": "https://ist.kpi.ua/wp-content/uploads/2021/09/polischuk.jpg",
    "Полторак Вадим Петрович": "https://ist.kpi.ua/wp-content/uploads/2021/09/poltorak.jpg",
    "Попенко Володимир Дмитрович": "https://ist.kpi.ua/wp-content/uploads/2021/09/popenko-1.jpg",
    "Резніков Сергій Анатолійович": "https://ist.kpi.ua/wp-content/uploads/2021/12/reznikov.jpg",
    "Рибачук Людмила Віталіївна": "https://ist.kpi.ua/wp-content/uploads/2021/09/rybachyk.jpg",
    "Савчук Олена Володимирівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/savchuk.jpg",
    "Сокульський Олег Євгенович": "https://ist.kpi.ua/wp-content/uploads/2021/09/sokulsky.jpg",
    "Сперкач Майя Олегівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/sperkach.jpg",
    "Тимошин Юрій Афанасійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/timoshin.jpg",
    "Ткач Михайло Мартинович": "https://ist.kpi.ua/wp-content/uploads/2021/09/tkach.jpg",
    "Ульяницька Ксенія Олександрівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/ulianytska.jpg",
    "Цеслів Ольга Володимирівна": "https://ist.kpi.ua/wp-content/uploads/2022/11/czesliv.jpg",
    "Шимкович Володимир Миколайович": "https://ist.kpi.ua/wp-content/uploads/2021/12/shymkovychv.jpg",
    "Анікін Володимир Костянтинович": "https://ist.kpi.ua/wp-content/uploads/2021/09/anikin.jpg",
    "Араффа Хальдун Осман": "https://ist.kpi.ua/wp-content/uploads/2021/12/araffa.jpg",
    "Базака Юрій Анатолійович": "https://ist.kpi.ua/wp-content/uploads/2021/12/bazakayua.jpg",
    "Баклан Ярослав Ігорович": "https://ist.kpi.ua/wp-content/uploads/2021/09/baklan.jpg",
    "Белоус Роман Володимирович": "https://ist.kpi.ua/wp-content/uploads/2021/12/belous.jpeg",
    "Бердник Юрій Михайлович": "https://ist.kpi.ua/wp-content/uploads/2021/09/berdnyk.jpg",
    "Вітюк Альона Євгеніївна": "https://ist.kpi.ua/wp-content/uploads/2021/12/vituk.jpg",
    "Вовк Євгеній Андрійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/vovk.jpg",
    "Галушко Дмитро Олександрович": "https://ist.kpi.ua/wp-content/uploads/2021/09/halushko.jpg",
    "Густера Олег Михайлович": "https://ist.kpi.ua/wp-content/uploads/2022/11/gustera.jpg",
    "Дорошенко Катерина Сергіївна": "https://ist.kpi.ua/wp-content/uploads/2021/09/doroshenko-kateryna.jpg",
    "Зубко Роман Анатолійович": "https://ist.kpi.ua/wp-content/uploads/2022/11/zubko.jpg",
    "Колеснік Валерій Миколайович": "https://ist.kpi.ua/wp-content/uploads/2022/12/kolesnik.jpg",
    "Майєр Ілля Сергійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/maiier.jpg",
    "Мітін Сергій Вячеславович": "https://ist.kpi.ua/wp-content/uploads/2021/09/mitin.jpg",
    "Моргаль Олег Михайлович": "https://ist.kpi.ua/wp-content/uploads/2021/09/morhal.jpg",
    "Нестерук Андрій Олександрович": "https://ist.kpi.ua/wp-content/uploads/2022/11/nesteruk.jpg",
    "Нікітін Валерій Андрійович": "https://ist.kpi.ua/wp-content/uploads/2022/11/nikitin.png",
    "Орленко Сергій Петрович": "https://ist.kpi.ua/wp-content/uploads/2021/12/orlenko.jpg",
    "Польшакова Ольга Михайлівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/polshakova.jpg",
    "Проскура Світлана Леонідівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/proskura.jpg",
    "Тимофєєва Юлія Сергіївна": "https://ist.kpi.ua/wp-content/uploads/2021/09/tymofieieva.jpg",
    "Хмелюк Володимир Сергійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/khmeliuk-volodymyr.jpg",
    "Хмелюк Марина Сергіївна": "https://ist.kpi.ua/wp-content/uploads/2021/09/khmeliuk-maryna.jpg",
    "Цимбал Святослав Ігорович": "https://ist.kpi.ua/wp-content/uploads/2021/09/tsymbal.jpg",
    "Шимкович Любов Леонідівна": "https://ist.kpi.ua/wp-content/uploads/2021/09/shymkovych.jpg",
    "Шинкевич Микола Костянтинович": "https://ist.kpi.ua/wp-content/uploads/2021/09/shynkevych.jpg",
    "Яланецький Валерій Анатолійович": "https://ist.kpi.ua/wp-content/uploads/2021/09/yalanetskyi.jpg"
}
ist_data_parsed = {}


class DescriptorData:
    def __init__(self, descriptor, shape, d):
        self.descriptor = descriptor
        self.shape = shape
        self.d = d


def get_descriptor(file, show_image):
    img = io.imread(file)
    if show_image:
        win1 = dlib.image_window()
        win1.clear_overlay()
        win1.set_image(img)

    dets = detector(img, 1)
    result = []
    for k, d in enumerate(dets):
        print(" Detection {}: Left : {} Top: {} Right : {} Bottom : {}".format(k, d.left(), d.top(), d.right(),
                                                                               d.bottom()))
        shape = sp(img, d)
        if show_image:
            win1.clear_overlay()
            win1.add_overlay(d)
            win1.add_overlay(shape)
            win1.wait_for_keypress("q")

        result.append(DescriptorData(facerec.compute_face_descriptor(img, shape), shape, d))

    return result


def download_image_from_url(url):
    print(url)
    image = io.imread(url)
    file_name = f'images/downloaded/{url.__hash__()}.jpeg'
    io.imsave(file_name, image)
    return file_name


def is_same_person(desc_1, desc_2):
    a = distance.euclidean(desc_1, desc_2)
    return a < 0.6


def task_one():
    desc_1 = get_descriptor("images/me.jpg", True)
    desc_2 = get_descriptor("images/camera.jpg", True)
    are_same = is_same_person(desc_1[0], desc_2[0])
    print(f"Is same person: {are_same}")


def extract_data():
    url = "https://ist.kpi.ua/uk/pedagogichnij-sklad/"
    page_data = requests.get(url).text
    data = BeautifulSoup(page_data, 'html.parser').find_all(class_="person__wrapper")
    for s in data:
        find = s.find(class_="person__name")
        if find is not None:
            name = find.string
            img = s.find("img")
            if img is not None:
                url = img.attrs["src"]
                if not url.endswith("png"):
                    ist_data_parsed[name] = url


def task_two():
    extract_data()

    for name, url in ist_data_parsed.items():
        img = download_image_from_url(url)
        descriptor = get_descriptor(img, False)
        if len(descriptor) > 0:
            Dictionary_Kafedra_[name + ", Кафедра Інформаційних Систем та Технологій"] = descriptor[0]

    print("Printing Dictionary_Kafedra_...")
    for key, value in Dictionary_Kafedra_.items():
        print(key, value)

    print("Looking in a photo...")
    url = "http://www.ntu.edu.ua/wp-content/uploads/2016/03/kist-2019.jpg"
    image_file = download_image_from_url(url)

    img = io.imread(image_file)
    win1 = dlib.image_window()
    win1.clear_overlay()
    win1.set_image(img)

    img_descriptors = get_descriptor(image_file, False)
    print(f"Found {len(img_descriptors)} people on the photo")
    for descriptor in img_descriptors:
        for key, descriptor_data in Dictionary_Kafedra_.items():
            if is_same_person(descriptor.descriptor, descriptor_data.descriptor):
                print(f"Found {key} in a photo")
                win1.clear_overlay()
                win1.add_overlay(descriptor.d)
                win1.add_overlay(descriptor.shape)
                win1.wait_for_keypress("q")


if __name__ == '__main__':
    sp = dlib.shape_predictor('dlib_files/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_files/dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()

    # task_one()
    task_two()
