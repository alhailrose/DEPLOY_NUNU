from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# with open('app/model.json', 'r') as json_file:
#     json_saved_model = json_file.read()

# model = tfjs.converters.load_keras_model(json_saved_model)
# model.load_weights('app/bin/')

model = load_model('./explorentt_mobilenet_v2.h5')

class_names = np.array([
    'Air Terjun Oenesu',
    'Air Terjun Tanggedu',
    'Air Terjun Tesbatan',
    'Bukit Wairinding',
    'Danau Kelimut',
    'Gua Rangko',
    'Lawa Darat Gili',
    'Lingko Spider Web Rice Field',
    'Pandar Island',
    'Pantai Koka',
    'Pantai Kolbano',
    'Pantai Mandorak',
    'Pantai Oetune ',
    'Pantai Waecicu',
    'Pantai Walakiri',
    'Pantai Watu Parunu',
    'Pink Beach',
    'Pulau Kalong',
    'Pulau Kanawa',
    'Rumah Budaya Sumba',
    'Savana Puru Kambera',
    'Taman Nostalgia Kupang',
    'Wae Rebo Village',
    'Waikuri Lagoon',
    'Wisata Adat Kampung Todo'])  

@app.get('/')
def read_root():
    return {'message': 'Image classification'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of an uploaded image.

    Args:
        file (UploadFile): An image file to predict.

    Returns:
        dict: A dictionary containing the predicted class.
    """
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    image = image.resize((150, 150))  
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  

    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    response = {'predicted_class': predicted_class}

    if predicted_class == 'Air Terjun Oenesu':
        response.update({
            'artikel': ("Air Terjun Oenesu adalah salah satu tempat wisata yang ada di Kabupaten Kupang, "
                        "Nusa Tenggara Timur yang terkenal akan keindahan alamnya. Air terjun ini adalah "
                        "air terjun bertingkat dengan aliran air yang jernih dan kolam natural yang disokong "
                        "oleh hutan hijau yang rimbun. Akses jalan menuju air terjun ini sangat mudah. Harga "
                        "tiket yang ditawarkan hanya 5.000 per orang. Oenesu sangat digemari oleh para pengendara "
                        "jalan untuk berenang dan bersantai karena suasana yang sangat tenang dan asri."),
            'lokasi': 'https://maps.app.goo.gl/hvtjpgSFLFtAQCAF7',
            'rating': 4.5
        })

    elif predicted_class == 'Air Terjun Tanggedu':
        response.update({
            'artikel': ("Air Terjun Tanggedu di Pulau Sumba, Nusa Tenggara Timur, Indonesia, menawarkan pemandangan alam "
                        "spektakuler dengan airnya yang jatuh dari tebing curam setinggi sekitar 100 meter. Terletak di "
                        "Kabupaten Waikabubak, air terjun ini dikelilingi oleh hutan tropis yang hijau. Perjalanan menuju "
                        "air terjun melibatkan pendakian dan trekking yang menantang namun sebanding dengan pemandangan "
                        "alam yang luar biasa. Bagi penggemar petualangan alam, Air Terjun Tanggedu adalah destinasi yang "
                        "sempurna untuk mengunjungi di Pulau Sumba."),
            'lokasi': 'https://maps.app.goo.gl/tukyxZEE1rTBHnLm9',
            'rating': 4.8
        })

    elif predicted_class == 'Air Terjun Tesbatan':
        response.update({
            'artikel': ("Air Terjun Tesbatan menawarkan pengalaman unik dengan tiga tingkatan yang berbeda. "
                        "Kolam di bawah air terjun memungkinkan pengunjung bermain air sambil menikmati keindahan "
                        "alam. Fasilitasnya meliputi lopo-lopo, area parkir, dan ruang ganti pakaian. Buka setiap "
                        "hari dari pukul 08.00 hingga 18.00, dengan harga karcis masuk yang terjangkau. Dengan segala "
                        "fasilitasnya, Tesbatan adalah tempat yang sempurna untuk bersantai dan menikmati alam."),
            'lokasi': 'https://maps.app.goo.gl/PnKwLQjRYZ4RVu9r9',
            'rating': 4.2
    })

    elif predicted_class == 'Bukit Wairinding':
        response.update({
            'artikel': ("Bukit Wairinding terletak di Desa Pambota Jara, Kecamatan Pandawai Sumba Timur, Nusa Tenggara Timur (NTT). "
                        "Daya tarik utama dari Bukit Wairinding tak lain adalah lanskap padang savana yang memanjakan mata. Sejauh mata "
                        "memandang, lekuk-lekuk bukit yang eksotis dan masih asri menjadikan lokasi ini begitu istimewa. Terlebih jika "
                        "waktu mendekati senja, garis cakrawala akan memunculkan warna yang memesona. Selain itu, udara juga tidak terlalu "
                        "panas sehingga wisatawan bisa lebih leluasa berjalan-jalan dan berfoto. Tak jarang wisatawan akan bertemu dengan "
                        "anak-anak Sumba yang tengah menggembalakan hewan ternaknya. Anak-anak Sumba ini juga kerap membantu menunjukkan jalan "
                        "bagi wisatawan. Tak jarang wisatawan mengajak mereka berfoto bersama, atau berfoto dengan kuda yang mereka bawa. "
                        "Suasana ini tentunya tidak bisa didapatkan di kota besar, sehingga akan menjadi pengalaman menarik dan berkesan yang "
                        "bisa diingat ketika pulang."),
            'lokasi': 'https://maps.app.goo.gl/8HYdtcx2wvz85M8XA',
            'rating': 4.8
    })

    elif predicted_class == 'Danau Kelimutu':
        response.update({
            'artikel': ("Gunung Kelimutu adalah gunung berapi yang terletak di Pulau Flores, Provinsi Nusa Tenggara Timur. "
                        "Lokasi gunung ini tepatnya di Desa Pemo, Kecamatan Kelimutu, Kabupaten Ende. Gunung ini memiliki "
                        "tiga buah danau kawah di puncaknya. Danau ini dikenal dengan nama Danau Tiga Warna karena memiliki "
                        "tiga warna yang berbeda, yaitu merah, biru, dan putih. Walaupun begitu, warna-warna tersebut selalu "
                        "berubah-ubah seiring dengan perjalanan waktu. Kelimutu merupakan gabungan kata dari 'keli' yang berarti "
                        "gunung dan kata 'mutu' yang berarti mendidih. Menurut kepercayaan penduduk setempat, warna-warna pada "
                        "danau Kelimutu memiliki arti masing-masing dan memiliki kekuatan alam yang sangat dahsyat. Luas ketiga "
                        "danau itu sekitar 1.051.000 meter persegi dengan volume air 1.292 juta meter kubik. Batas antar danau adalah "
                        "dinding batu sempit yang mudah longsor. Dinding ini sangat terjal dengan sudut kemiringan 70 derajat. "
                        "Ketinggian dinding danau berkisar antara 50 sampai 150 meter."),
            'lokasi': 'https://maps.app.goo.gl/BLUBGdrdRpC5oR2r5',
            'rating': 4.7
        })

    elif predicted_class == 'Gua Rangko':
        response.update({
            'artikel': ("Gua Rangko adalah tempat wisata yang terletak di Pulau Gusung, Desa Rangko, Kecamatan Boleng, Kabupaten Manggarai "
                        "Barat, Nusa Tenggara Timur (NTT). Tempat ini menawarkan keindahan gua alami dengan stalaktit khas gua. Di dalamnya, "
                        "pengunjung akan melihat kolam alam yang sangat jernih sehingga batuan stalakmit di dasar kolam terlihat jelas. Jika "
                        "terkena sinar matahari, kolam dengan kedalaman empat meter itu akan terlihat semakin indah. Pengunjung juga dapat "
                        "berenang di kolam ini untuk menikmati kesegaran air alam, namun mereka diminta untuk berhati-hati agar anggota badan "
                        "tidak terkena stalakmit yang terdapat di sekitar kolam."),
            'lokasi': 'https://maps.app.goo.gl/zKNTzGf3r9wgNf5f9',
            'rating': 4.5
        })

    elif predicted_class == 'Lawa Darat Gili':
        response.update({
            'artikel': ("Gili Lawa Darat adalah salah satu pulau kecil yang terletak di Taman Nasional Komodo, di Provinsi Nusa Tenggara "
                        "Timur, Indonesia. Pulau ini terkenal karena pemandangan alamnya yang menakjubkan, termasuk bukit-bukit kecil, "
                        "hamparan padang rumput, dan tebing-tebing batu yang menjulang di sepanjang pantainya. Gili Lawa Darat juga "
                        "merupakan tempat yang populer untuk aktivitas snorkeling dan diving karena terumbu karangnya yang indah dan "
                        "keanekaragaman hayati bawah laut yang kaya. Pemandangan matahari terbenam dari pulau ini juga sangat terkenal, "
                        "memberikan pengalaman yang luar biasa bagi para pengunjung. Keindahan alamnya yang menakjubkan menjadikan Gili "
                        "Lawa Darat sebagai salah satu tujuan wisata yang sangat dicari di wilayah tersebut."),
            'lokasi': 'https://maps.app.goo.gl/zEZCTyYzyE1eEPWq6',
            'rating': 4.9
        })

    elif predicted_class == 'Lingko Spider Web Rice Field':
        response.update({
            'artikel': ("Lingko Spider Web Rice Field adalah istilah yang mengacu pada pola ladang padi tradisional yang terdapat di "
                        "Sumba, Nusa Tenggara Timur, Indonesia. Ladang-ladang ini terkenal karena polanya yang menyerupai jaring "
                        "laba-laba atau spider web ketika dilihat dari udara. Mereka adalah contoh unik dari sistem penanaman "
                        "terasering tradisional yang masih dipraktikkan oleh masyarakat Sumba. Pola ini tidak hanya memiliki nilai "
                        "fungsional dalam pengelolaan air dan tanah, tetapi juga memiliki makna simbolis dan budaya yang dalam bagi "
                        "masyarakat lokal. Lingko Spider Web Rice Field adalah contoh menakjubkan dari keterampilan tradisional dalam "
                        "pertanian yang telah bertahan selama berabad-abad di Pulau Sumba."),
            'lokasi': 'https://maps.app.goo.gl/dpkiwxWwtcEJJnRH9',
            'rating': 4.3
    })

    elif predicted_class == 'Padar Island':
        response.update({
            'artikel': ("Padar Island adalah salah satu pulau yang terletak di Taman Nasional Komodo, Nusa Tenggara Timur, Indonesia. "
                        "Pulau ini terkenal dengan pemandangan alamnya yang spektakuler, terutama panorama dari puncak bukitnya yang "
                        "menawarkan pemandangan pantai dengan air berwarna biru dan pasir putih."),
            'lokasi': 'https://maps.app.goo.gl/p9H8cM37AEEW1B4m9',
            'rating': 4.8
    })

    elif predicted_class == 'Pantai Koka':
        response.update({
            'artikel': ("Pantai Koka, juga dikenal sebagai Paga Beach, adalah salah satu tempat paling indah di Flores. Pantai ini memiliki "
                        "pemandangan yang memukau dengan pasir putih yang dikelilingi oleh tebing karang yang ditumbuhi flora tropis. Pantai "
                        "ini adalah pilihan sempurna untuk liburan pantai klasik, menawarkan suasana yang sempurna untuk piknik, pertemuan "
                        "romantis, dan jalan-jalan santai di malam hari di sepanjang pantai. Hembusan angin laut yang segar memberikan sensasi "
                        "yang menyenangkan sepanjang hari, sementara air biru yang jernih, dengan pendekatan yang lembut dan dasar laut "
                        "berpasir, menjamin kondisi yang ideal untuk berenang."),
            'lokasi': 'https://maps.app.goo.gl/P7XJdCq6bHj7twwx9',
            'rating': 4.4
    })

    elif predicted_class == 'Pantai Kolbano':
        response.update({
            'artikel': ("Pantai Kolbano adalah sebuah destinasi wisata yang terletak di Desa Kolbano, Kecamatan Kolbano, Kabupaten Timor "
                        "Tengah Selatan, Nusa Tenggara Timur. Pantai ini dikenal dengan keunikannya, yaitu tidak memiliki hamparan pasir "
                        "seperti pantai pada umumnya, melainkan batuan. Pantai ini menghadap ke Samudera Hindia dan memiliki bibir pantai "
                        "dengan hamparan batu licin berwarna-warni. Salah satu ciri khas dari pantai ini adalah adanya batu raksasa yang "
                        "disebut ‘Fatu Un’, yang jika dilihat dari samping, batu ini mirip kepala singa."),
            'lokasi': 'https://maps.app.goo.gl/Jvru1Qp3sDQALoL18',
            'rating': 4.5
    })

    elif predicted_class == 'Pantai Mandorak':
        response.update({
            'artikel': ("Pantai Mandorak adalah sebuah destinasi wisata yang terletak di Desa Kalena Rongo, Kecamatan Kodi Utara, "
                        "Kabupaten Sumba Barat Daya, Nusa Tenggara Timur. Pantai ini dikenal dengan pasir putihnya dan batu karang. "
                        "Suasana pantai makin terasa alami dengan deburan ombak yang keras. Batu karang Pantai Mandorak saling berhimpit "
                        "membuat pandangan ke pantai cukup sempit karena tertutup tebing. Pantai ini memiliki pesona alam yang memikat "
                        "dengan pasir putih serta air laut yang jernih berwarna hijau toska. Pantai Mandorak juga dikelilingi oleh "
                        "perbukitan yang menambah keindahan alamnya. Pantai Mandorak dikenal dengan gelombang lautnya yang tinggi, "
                        "sehingga lebih cocok untuk bermain-main di air daripada berenang."),
            'lokasi': 'https://maps.app.goo.gl/2ZazDuVvtrfgUkHW',
            'rating': 4.5
    })

    elif predicted_class == 'Pantai Oetune':
        response.update({
            'artikel': ("Pantai Oetune merupakan pantai yang sangat unik dan indah di Nusa Tenggara Timur (NTT). Pasalnya Pantai Oetune "
                        "memiliki pasir yang halus seperti pasir gurun. Tak hanya pasirnya yang halus pantai ini pun memiliki dentuman "
                        "ombak yang sangat kencang dan pecah di setiap gulungannya. Di tepi pantai ini pun dikelilingi banyaknya Pohon "
                        "Kasuari yang berusia puluhan tahun. Pantai Oetune ini berlokasi di Tuafanu, Kecamatan Kualin, Kabupaten Timor "
                        "Tengah Selatan, NTT. Akses menuju Pantai Oetune kurang lebih 115 kilometer dari pusat Kota Kupang dan jarak "
                        "tempuh sekitar 2-3 jam di perjalanan. Air laut yang ada di Pantai Oetune ini sangat biru dan bersih hingga Anda "
                        "dapat bercermin. Pantai Oetune memiliki pasir putih yang sangat halus dan menggunung membuat seolah-olah sedang "
                        "berada di gurun pasir."),
            'lokasi': 'https://maps.app.goo.gl/AxuQUF5Nd7BPkTXd9',
            'rating': 4.6
    })

    elif predicted_class == 'Pantai Waecicu':
        response.update({
            'artikel': ("Pantai Waecicu adalah sebuah destinasi wisata yang terletak di Labuan Bajo, Komodo, Kabupaten Manggarai Barat, "
                        "Nusa Tenggara Timur. Pantai ini dikenal dengan pasir putihnya yang membentang sepanjang 1 km dan banyak bukit "
                        "hijau kecil di sekitarnya. Pantai ini memiliki tipe perairan yang dangkal dan berbagai pemandangan biota laut "
                        "yang cantik. Selain itu, Pantai Waecicu juga merupakan spot terbaik untuk menikmati matahari terbenam (sunset) "
                        "sambil duduk santai di jembatan yang digunakan oleh nelayan bersandar perahu."),
            'lokasi': 'https://maps.app.goo.gl/tdp3Ac6wTDxEFHjZA',
            'rating': 3.5
    })

    elif predicted_class == 'Pantai Walakiri':
        response.update({
            'artikel': ("Pantai Walakiri adalah surga pasir putih yang tersembunyi di Pulau Sumba, Nusa Tenggara Timur, Indonesia. "
                        "Terletak di Desa Walakiri, Kecamatan Kodi, pantai ini menawarkan pemandangan yang menakjubkan dengan pasir "
                        "putihnya yang halus, air laut yang biru jernih, dan batu-batu karang yang menjulang di sekitarnya. Salah "
                        "satu daya tarik utama Pantai Walakiri adalah pohon lontar yang ikonik yang tumbuh di sepanjang pantai dan "
                        "menjadi spot foto yang populer bagi para pengunjung. Selain menikmati keindahan alam, pengunjung juga dapat "
                        "bersantai, berenang, atau berjemur di Pantai Walakiri. Dengan suasana yang tenang dan keindahan alam yang "
                        "memukau, Pantai Walakiri adalah destinasi yang sempurna untuk melarikan diri dari keramaian dan menikmati "
                        "pesona alam Sumba yang masih alami."),
            'lokasi': 'https://maps.app.goo.gl/WSkR9tjEgAyRjbuj8',
            'rating': 4.6
    })

    elif predicted_class == 'Pantai Watu Parunu':
        response.update({
            'artikel': ("Pantai Watu Parunu adalah destinasi wisata yang menawarkan pantai, perbukitan, dan tebing. Pantai ini "
                        "memiliki garis pantai yang sangat panjang dan berhubungan langsung dengan Samudra Indonesia, sehingga "
                        "ombaknya lumayan besar. Pengunjung yang datang ke Pantai Watu Parunu akan disambut hamparan pasir putih, "
                        "tebing batu, dan desir ombak yang datang silih berganti. Tebing batu di Pantai Watu Parunu berwarna "
                        "putih dan berada di ujung timur, menjadi salah satu ikon dari Pantai Watu Parunu yang selalu menjadi "
                        "spot untuk berfoto. Nama Watu Parunu sendiri berkaitan dengan fenomena alam batu berlubang yang menjadi "
                        "ciri khas lain dari pantai ini. Untuk bisa berjalan melewati batu berlubang ini, pengunjung harus merunduk."),
            'lokasi': 'https://maps.app.goo.gl/zaRNyMpVksxwfPJDA',
            'rating': 4.5
    })

    elif predicted_class == 'Pink Beach':
        response.update({
            'artikel': ("Pink Beach adalah salah satu pantai eksotis yang memiliki pemandangan indah dengan pasir pantai berwarna "
                        "merah muda atau pink. Pantai ini juga dikenal dengan nama Pantai Merah di Pulau Komodo. Melansir dari "
                        "Labuanbajotour, Pink Beach mempunyai warna pasir yang unik dan kerap menjadi destinasi wisata favorit "
                        "turis lokal dan mancanegara. Diketahui, warna pink pada pasirnya berasal dari hewan berukuran mikroskopis "
                        "bernama “Foraminifera” yang memberikan pigmen warna merah pada koral. Sehingga, koral-koral tersebut "
                        "terbawa oleh gelombang dan berakhir di pesisir pantai dan hancur menjadi serpihan serta butiran yang menjadi "
                        "pasir pantai. Sehingga, pantai ini pun disebut Pantai Pink yang memesona."),
            'lokasi': 'https://maps.app.goo.gl/JQCSuboXnnrAcS8p8',
            'rating': 4.8
    })

    elif predicted_class == 'Pulau Kalong':
        response.update({
            'artikel': ("Pulau Kalong, juga dikenal sebagai Kalong Padar atau Flying Fox Island, adalah sebuah pulau kecil yang "
                        "tidak berpenghuni dan terletak di Taman Nasional Komodo, dekat Labuan Bajo di provinsi Nusa Tenggara "
                        "Timur, Indonesia. Pulau ini dinamai setelah ribuan kelelawar yang menghuninya. Pulau ini menjadi salah "
                        "satu destinasi yang dikenal memiliki koloni kelelawar dengan jumlah besar yang mendiami pohon bakau di "
                        "pulau Kalong. Pulau ini adalah rumah bagi ribuan satwa liar yang masih eksis dan menggantungkan hidupnya "
                        "di sana, termasuk monyet, burung, hingga reptil. Selain itu, Pulau Kalong juga memiliki berbagai jenis "
                        "terumbu karang yang indah."),
            'lokasi': 'https://maps.app.goo.gl/zn1URtEZLpCAo1Ay9',
            'rating': 5
    })

    elif predicted_class == 'Pulau Kanawa':
        response.update({
            'artikel': ("Pulau Kanawa adalah sebuah pulau kecil yang terletak di perairan Flores, Nusa Tenggara Timur (NTT). "
                        "Pulau ini sangat terkenal karena keindahan bawah lautnya yang memukau. Ada banyak sekali terumbu "
                        "karang yang tumbuh di pulau ini dan itu menjadi daya tarik bagi wisatawan yang suka menyelam "
                        "sekaligus melihat kecantikan bawah laut Pulau Kanawa. Pulau ini memiliki luas sekitar 32 ha dan "
                        "jaraknya kurang lebih sekitar 15 km dari Labuan Bajo. Banyak yang bilang Pulau Kanawa Labuan Bajo "
                        "adalah gerbang dari Pulau Komodo. Pulau ini sangat cocok untuk kamu yang tidak "
                        "suka dengan tempat ramai."),
            'lokasi': 'https://maps.app.goo.gl/XphhXGxR5aqTDbRf9',
            'rating': 4
    })

    elif predicted_class == 'Rumah Budaya Sumba':
        response.update({
            'artikel': ("Rumah Budaya Sumba adalah museum khusus yang digunakan untuk memperkenalkan sejarah dan budaya "
                        "Sumba. Fungsi Rumah Budaya Sumba adalah sebagai museum sekaligus tempat wisata, penelitian, dan "
                        "pertemuan, serta pusat pembelajaran kebudayaan Sumba. Rumah Budaya Sumba mengoleksi berbagai "
                        "macam peninggalan kelompok etnik daerah Sumba yang berasal dari masa prasejarah hingga masa kini. "
                        "Koleksi-koleksi ini merupakan sumbangan koleksi pribadi Pater Robert Ramone dan sumbangan dari "
                        "setiap rumah adat Sumba. Rumah Budaya Sumba terletak di Jalan Rumah Budaya Nomor 212, Kalembu "
                        "Nga’banga Waitabula, Kabupaten Sumba Barat Daya, Provinsi Nusa Tenggara Timur."),
            'lokasi': 'https://maps.app.goo.gl/oKVbDfiYY7QPHAtP6',
            'rating': 3.5
    })

    elif predicted_class == 'Savana Puru Kambera':
        response.update({
            'artikel': ("Savana Puru Kambera di Sumba Timur, Nusa Tenggara Timur, adalah rumah bagi kuda liar Sumba "
                        "dan menawarkan padang rumput yang luas dengan panorama indah. Terletak dekat dengan Pantai "
                        "Puru Kambera, sabana ini menjadi destinasi utama bagi pengunjung yang ingin melihat gerombolan "
                        "kuda liar. Dengan pemandangan ala Afrika dan latar belakang biru laut, Sabana Puru Kambera adalah "
                        "tempat yang menakjubkan untuk dinikmati."),
            'lokasi': 'https://maps.app.goo.gl/4ehKM9xiFN2rRMiP7',
            'rating': 4.6
    })

    elif predicted_class == 'Taman Nostalgia Kupang':
        response.update({
            'artikel': ("Taman Nostalgia Kupang adalah sebuah tempat yang unik dan bersejarah. Taman ini dikenal karena "
                        "menjadi rumah bagi Gong Perdamaian, sebuah gong raksasa yang menjadi simbol perdamaian dan "
                        "persatuan. Gong ini merupakan hadiah dari Pemerintah Tiongkok kepada Provinsi Nusa Tenggara Timur. "
                        "Selain Gong Perdamaian, taman ini juga memiliki replika bangunan-bangunan khas Nusa Tenggara Timur "
                        "dan area hijau yang luas. Pengunjung dapat menikmati suasana yang tenang dan berfoto di sekitar "
                        "taman. Taman Nostalgia Gong Perdamaian menjadi destinasi yang menarik bagi wisatawan yang ingin "
                        "menikmati keindahan alam dan belajar tentang budaya lokal."),
            'lokasi': 'https://maps.app.goo.gl/2HFMWe9NRaBPAJR39',
            'rating': 4.4
    })

    elif predicted_class == 'Wae Rebo Village':
        response.update({
            'artikel': ("Wae Rebo adalah desa adat kecil yang terletak di pegunungan terpencil, Kampung Satar Lenda, "
                        "Kecamatan Satar Mese Barat, Kabupaten Manggarai, Nusa Tenggara Timur. Desa ini dikenal dengan 7 "
                        "rumah adatnya dan pemandangan khas pegunungan."),
            'lokasi': 'https://maps.app.goo.gl/oYo4uGq4E3ApMavy5',
            'rating': 4.7
    })

    elif predicted_class == 'Waikuri Lagoon':
        response.update({
            'artikel': ("Waikuri Lagoon hadir dengan air yang sangat jernih dan berwarna biru kehijauan dengan kandungan "
                        "air asin dan payau. Ya, tidak seperti danau lainnya, air danau ini asin karena terbentuk dari "
                        "air lautan lepas yang ada di sekitar danau dan masuk dari celah bebatuan di gugusan karang "
                        "sekitarnya. Laguna ini dipagari batu karang alami di mana tumbuh pepohonan rimbun dan semak "
                        "belukar yang seolah menyembunyikan keindahannya dari kejauhan. Selain bermain di laguna, para "
                        "pengunjung juga bisa menyusuri jalan setapak ke arah barat dan mendaki bukit karang yang menjorok "
                        "ke lautan lepas dengan hati-hati. Dari atas bukit karang ini, pengunjung dapat menikmati pesona "
                        "Danau Weekuri dari ketinggian dan ombak lautan biru di sekitarnya yang menghantam gugusan karang "
                        "di sekitar danau."),
            'lokasi': 'https://maps.app.goo.gl/8MGi7iAQLrChGYen9',
            'rating': 4.7
    })

    elif predicted_class == 'Wisata Adat Kampung Todo':
        response.update({
            'artikel': ("Kampung Adat Todo adalah sebuah kampung adat yang terletak di Desa Todo, Kecamatan Satar Mese, "
                        "Kabupaten Manggarai, Pulau Flores, Provinsi Nusa Tenggara Timur. Kampung ini dikenal sebagai kampung "
                        "adat tertua di Manggarai Raya dan memiliki bentuk bangunan rumah adat yang khas serta lingkungan "
                        "perkampungan adat yang indah. Kampung Adat Todo juga dikenal sebagai pusat peradaban Minangkabau. "
                        "Selain itu, kampung ini juga merupakan pusat kerajaan Manggarai di zaman dulu. Di sini, wisatawan "
                        "dapat melakukan berbagai aktivitas seperti merasakan bagaimana tamu disambut dengan cara adat orang "
                        "Manggarai, mengenakan pakaian adat keseharian masyarakat Todo, belajar bertenun dengan ibu-ibu yang "
                        "menenun di depan rumah serta belajar sejarah mengenai kampung Todo dan kerajaan Todo."),
            'lokasi': 'https://maps.app.goo.gl/GXqcJSFkqFC9q9Xd8',
            'rating': 3.5
    })

    return response
