"""Importowanie wszystkich bibliotek"""
from pylab import *
from pymap3d import ecef2geodetic, ecef2aer, aer2geodetic
import georinex as gr
import pandas as pd
from datetime import datetime
import numpy as np
from pyunpack import Archive
import shutil
import urllib.request, urllib.error
from contextlib import closing
import string
import glob
import os, sys
import itertools

"""Stałe wartości"""
c0 = 3E8
c = 2997

def downloadFile_Galileo():

    """obecna data i godzina"""
    now = datetime.now()

    """wybieranie najnowszego pliku z folderu i wyznaczanie daty jego pobrania"""
    list_of_files = glob.glob('ionosphere_storm\\IGS\\navigations_file\\Galileo\\*')
    print(list_of_files)

    latest_file = max(list_of_files, key=os.path.getctime)
    # datetime_object = datetime.strptime(time.ctime(os.path.getmtime(latest_file)), '%a %b %d %H:%M:%S %Y')

    """różnica miedzy obecną datą, a datą pobrania najnowszego pliku -> jeżeli różnica jest większa niż 1 godzina pobiera nowy plik"""
    # roznica = now - datetime_object
    # dt_roznica = roznica.seconds // 3600
    #
    day_of_year = datetime.now().strftime('%j')
    year = datetime.now().year
    # #
    # if dt_roznica > 1:
    url2 = 'ftp://igs.bkg.bund.de/IGS/obs/' + str(year) + '/' + str(day_of_year) + '/' + 'WROC00POL_R_' + str(year) + str(day_of_year) + '0000_01D_EN.rnx.gz'

    while True:
        try:
            with closing(urllib.request.urlopen(url2)) as r:
                with open(('WROC00POL_R_' + str(year) + str(day_of_year) + '0000_01D_EN.rnx.gz'), 'wb') as f:
                    shutil.copyfileobj(r, f)
            break
        except urllib.error.HTTPError as e:
            # Return code error (e.g. 404, 501, ...)
            print('HTTPError: {}'.format(e.code))
        except urllib.error.URLError as e:
            # Not an HTTP-specific error (e.g. connection refused)
            print('URLError: {}'.format(e.reason))
        day_of_year = int(day_of_year) - 1
        url2 = 'ftp://igs.bkg.bund.de/IGS/obs/' + str(year) + '/' + str(day_of_year) + '/' + 'WROC00POL_R_' + str(year) + str(day_of_year) + '0000_01D_EN.rnx.gz'

    # POBIERANIE PLIKU Z SERWERA FTP
    with closing(urllib.request.urlopen(url2)) as r:
        with open(('WROC00POL_R_' + str(year) + str(day_of_year) + '0000_01D_EN.rnx.gz'), 'wb') as f:
            shutil.copyfileobj(r, f)

    # DEKOPRESJA PLIKU
    Archive('WROC00POL_R_' + str(year) + str(day_of_year) + '0000_01D_EN.rnx.gz').extractall(
            'ionosphere_storm/IGS/navigations_file/Galileo')

    from pathlib import Path
    paths = sorted(Path('ionosphere_storm\\IGS\\navigations_file\\Galileo').iterdir(), key=os.path.getmtime)[:-1]
    for path in paths:
        os.remove(path)

    return

def downloadFile(type):

    # WYBIERANIE PLIKU NAJBLIŻSZEGO OBECNEJ DACIE
    day_of_year = datetime.now().strftime('%j')
    current_hour = datetime.now().strftime('%H')
    hour = datetime.now().hour
    year = datetime.now().year
    year = str(year)[-2:]
    # NAZWA STACJI
    station = 'wroc'

    # SŁOWNIK
    keys = [i for i in range(24)]
    value = list(string.ascii_lowercase[0:24])
    dictionary = dict(zip(keys, value))

    """warunek dla plików młodszych niż 1 godzina -> skrót myślowy"""

    """obecna data i godzina"""
    now = datetime.now()

    """wybieranie najnowszego pliku z folderu i wyznaczanie daty jego pobrania"""
    if type == 'd':
        list_of_files = glob.glob('ionosphere_storm\\IGS\\observations_file\\*')
    else:
        list_of_files = glob.glob('ionosphere_storm\\IGS\\navigations_file\\GPS\\*')

    latest_file = max(list_of_files, key=os.path.getctime)
    # datetime_object = datetime.strptime(time.ctime(os.path.getmtime(latest_file)), '%a %b %d %H:%M:%S %Y')
    #
    # """różnica miedzy obecną datą, a datą pobrania najnowszego pliku -> jeżeli różnica jest większa niż 1 godzina pobiera nowy plik"""
    # roznica = now - datetime_object
    # dt_roznica = roznica.seconds // 3600
    #
    # if dt_roznica > 1:

    # WARUNEK GDY BRAK PLIKU

    url = 'ftp://igs.bkg.bund.de/IGS/nrt/' + day_of_year + '/' + current_hour + '/' + station + day_of_year + str(
        dictionary.get(hour)) + '.' + str(year) + str(type) + '.gz'

    while True:
        try:
            conn = urllib.request.urlopen(url)
            with closing(urllib.request.urlopen(url)) as r:
                with open(
                        (station + str(day_of_year) + str(dictionary.get(hour)) + '.' + str(year) + str(type) + '.gz'),
                        'wb') as f:
                    shutil.copyfileobj(r, f)
            break
        except urllib.error.HTTPError as e:
            # Return code error (e.g. 404, 501, ...)
            print('HTTPError: {}'.format(e.code))
        except urllib.error.URLError as e:
            # Not an HTTP-specific error (e.g. connection refused)
            print('URLError: {}'.format(e.reason))
        current_hour = int(current_hour) - 1
        if 0 <= current_hour < 10:
            current_hour = '0' + str(current_hour)
        hour = hour - 1
        url = 'ftp://igs.bkg.bund.de/IGS/nrt/' + str(day_of_year) + '/' + str(
            current_hour) + '/' + station + str(day_of_year) + str(dictionary.get(hour)) + '.' + str(year) + str(
            type) + '.gz'
        if hour == -1:
            day_of_year = int(day_of_year) - 1
            if day_of_year < 100 and day_of_year > 9:
                day_of_year = '0' + str(day_of_year)
            elif day_of_year < 10 and day_of_year > -1:
                day_of_year = '00' + str(day_of_year)
            current_hour = 23
            hour = 23
            url = 'ftp://igs.bkg.bund.de/IGS/nrt/' + str(day_of_year) + '/' + str(current_hour) + '/' + station + str(
                day_of_year) + str(dictionary.get(hour)) + '.' + str(year) + str(type) + '.gz'

    # POBIERANIE PLIKU Z SERWERA FTP
    with closing(urllib.request.urlopen(url)) as r:
        with open((station + str(day_of_year) + str(dictionary.get(hour)) + '.' + str(year) + str(type) + '.gz'),
                  'wb') as f:
            shutil.copyfileobj(r, f)

    # DEKOPRESJA PLIKU
    if type == 'd':
        Archive(
            station + str(day_of_year) + str(dictionary.get(hour)) + '.' + str(year) + str(type) + '.gz').extractall(
            'ionosphere_storm/IGS/observations_file')
    if type == 'n':
        Archive(
            station + str(day_of_year) + str(dictionary.get(hour)) + '.' + str(year) + str(type) + '.gz').extractall(
            'ionosphere_storm/IGS/navigations_file/GPS')

    from pathlib import Path
    paths_obs = sorted(Path('ionosphere_storm\\IGS\\observations_file').iterdir(), key=os.path.getmtime)[:-1]
    for path in paths_obs:
        os.remove(path)
    paths_nav = sorted(Path('ionosphere_storm\\IGS\\navigations_file\\GPS').iterdir(), key=os.path.getmtime)[:-1]
    for path in paths_nav:
        os.remove(path)

    return


def latest_file(type="O", system="GPS"):
    if type == "N" and system == "GPS":
        list_of_files = glob.glob(
            'ionosphere_storm\\IGS\\navigations_file\\GPS\\*')
        latest_file = max(list_of_files, key=os.path.getctime)
        file = latest_file
    elif type == "N" and system == "Galileo":
        list_of_files = glob.glob(
            'ionosphere_storm\\IGS\\navigations_file\\Galileo\\*')
        latest_file = max(list_of_files, key=os.path.getctime)
        file = latest_file
    else:
        list_of_files = glob.glob(
            'ionosphere_storm\\IGS\\observations_file\\*')
        latest_file = max(list_of_files, key=os.path.getctime)
        file = latest_file
    return file


def getRangeSlantTEC(system="GPS"):
    """Funkcja zwraca ukośną wartość TEC w jednostkach TECU,
    obliczoną na podstawie obserwacji kodowych"""

    """lista wszystkich plików znajdujących się w podanym folderze, 
    wybrany zostaje plik ostatnio pobrany"""

    file = latest_file("O")

    if system == "GPS":
        # częstotliwości dla systemu GPS
        f1 = 1575.42E6
        f2 = 1227.6E6
        f5 = 1176.45E6

        # wczytywanie pliku obserwacyjnego i obserwacji dla systemu GPS

        obs = gr.load((str(file)), use='G')

        obs_data = obs.to_dataframe()
        df_obs = obs_data.reset_index()

        # lista satelitów GPS
        sv_list = df_obs['sv'].tolist()
        sv_list = list(set(sv_list))
        sv_list.sort()

        # sprawdzanie obserwacje na jakich częstotliwościach wybrać
        c_1 = []
        c_2 = []
        p_2 = []
        c_5 = []
        sv_ = []

        dates = []

        times_64 = getSTECtimes()
        # rzutowanie listy wartości do tablicy
        for sv in sv_list:
            temp = df_obs.loc[df_obs['sv'] == sv]

            C1 = temp['C1'].tolist()
            c_1.append(C1)

            C2 = temp['C2'].tolist()
            c_2.append(C2)

            P2 = temp['P2'].tolist()
            p_2.append(P2)

            C5 = temp['C5'].tolist()
            c_5.append(C5)

            sv_.append([sv] * len(C1))
        dates.append(times_64 * len(sv_list))

        # sv_ = list(itertools.chain.from_iterable(sv_))
        dates = list(itertools.chain.from_iterable(dates))
        cc_1 = list(itertools.chain.from_iterable(c_1))
        cc_2 = list(itertools.chain.from_iterable(c_2))
        pp_2 = list(itertools.chain.from_iterable(p_2))
        cc_5 = list(itertools.chain.from_iterable(c_5))

        # tworzenie df z wartościami obserwacji z podziałem na konkretne satelity
        df = pd.DataFrame([cc_1, cc_2, pp_2, cc_5]).transpose().rename(columns={0: 'C1', 1: 'C2', 2: 'P2', 3: 'C5'})
        count_nans = df.isna().sum().sort_values(ascending=True)
        two_col = (count_nans.iloc[0:2].index).tolist()
        two_col = sorted(two_col)

        lista_C2P2 = ['C2', 'P2']

        if set(two_col) == set(lista_C2P2):
            two_col = ['C1', 'C2']

        result = df[two_col]

        column_names = result.columns.tolist()
        result = result.rename(columns={column_names[0]: str(column_names[0]), column_names[1]: str(column_names[1])})

        result['time_only'] = [x.strftime('%H:%M:%S') for x in dates]
        result['sv'] = df_obs['sv']

    else:
        # częstotliwości dla systemu Galileo
        f1 = 1575.42E6
        f2 = 1278.75E6
        f5 = 1191.795E6

        # wczytywanie pliku obserwacyjnego i obserwacji dla systemu Galileo
        obs = gr.load((str(file)), use='E')
        obs_data = obs.to_dataframe()
        df_obs = obs_data.reset_index()

        # lista satelitów GPS
        sv_list = df_obs['sv'].tolist()
        sv_list = list(set(sv_list))
        sv_list.sort()

        # sprawdzanie obserwacje na jakich częstotliwościach wybrać
        c_1 = []
        c_2 = []
        c_5 = []
        sv_ = []

        dates = []

        times_64 = getSTECtimes()
        # rzutowanie listy wartości do tablicy
        for sv in sv_list:
            temp = df_obs.loc[df_obs['sv'] == sv]

            C1 = temp['C1'].tolist()
            c_1.append(C1)

            C2 = temp['C2'].tolist()
            c_2.append(C2)

            C5 = temp['C5'].tolist()
            c_5.append(C5)

            sv_.append([sv] * len(C1))
        dates.append(times_64 * len(sv_list))

        # sv_ = list(itertools.chain.from_iterable(sv_))
        dates = list(itertools.chain.from_iterable(dates))
        cc_1 = list(itertools.chain.from_iterable(c_1))
        cc_2 = list(itertools.chain.from_iterable(c_2))
        cc_5 = list(itertools.chain.from_iterable(c_5))

        # tworzenie df z wartościami obserwacji z podziałem na konkretne satelity
        df = pd.DataFrame([cc_1, cc_2, cc_5]).transpose().rename(columns={0: 'C1', 1: 'C2', 2: 'C5'})
        count_nans = df.isna().sum().sort_values(ascending=True)
        two_col = (count_nans.iloc[0:2].index).tolist()
        two_col = sorted(two_col)
        result = df[two_col]

        column_names = result.columns.tolist()
        result = result.rename(columns={column_names[0]: str(column_names[0]), column_names[1]: str(column_names[1])})

        result['time_only'] = [x.strftime('%H:%M:%S') for x in dates]
        result['sv'] = df_obs['sv']

    # dobieranie odpowiedniej częstotliwości
    dict_frequency = {'C1': f1, 'C2': f2, 'P2' : f2, 'C5': f5}
    f1 = dict_frequency[two_col[0]]
    f2 = dict_frequency[two_col[1]]

    sTEC = (pow(f1, 2) * pow(f2, 2) * (result.iloc[:, 0] - result.iloc[:, 1])) / (
            40.3 * (pow(f2, 2) - pow(f1, 2))) / pow(10, 16)
    result['roznica'] = result.iloc[:, 1] - result.iloc[:, 0]
    result['sTEC'] = sTEC

    return result

def getSTECtimes():
    """Funkcja zwraca listę epok dla wartości STEC z jednego pliku RINEX"""
    file = latest_file("O")
    obs = gr.load((str(file)), use='G')
    times = obs.time.to_dataframe()
    times = times['time'].astype('datetime64[s]')
    times_64 = pd.to_datetime(times).tolist()
    return times_64

def date_and_time():
    # times = getTimes()
    # times = times[0]
    # times = times.astype('datetime64[s]')
    times_64 = getSTECtimes()
    times_64 = times_64[0]
    data = str(times_64.date())
    godzina = str(times_64.hour)
    minuta = str(times_64.minute)
    sekundy = str(times_64.second)
    if len(minuta) < 2:
        minuta = '0' + minuta
    if len(sekundy) < 2:
        sekundy = '0' + sekundy
    return data, godzina, minuta, sekundy

def getTimes(file):
    """Zwraca listę czasów, na które zarejestrowano obserwacje"""
    times = file.time.to_dataframe()
    dt = list(times['time'].values)  # data w formacie numpy.datetime64
    return dt

def getGpsTime(dt):
    """Zamienia czas z formatu numpy.datetime64 na czas GPS"""
    # zamiana z formatu numpy.datetime64 na datetime.datetime
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (dt - unix_epoch) / one_second
    dt = datetime.utcfromtimestamp(seconds_since_epoch)
    # obliczenie czasu GPS
    gpsTime = 0
    days = (dt.weekday() + 1) % 7  # this makes Sunday = 0, Monday = 1, etc.
    gpsTime += days * 3600 * 24
    gpsTime += dt.hour * 3600
    gpsTime += dt.minute * 60
    gpsTime += dt.second
    return (gpsTime)

def condition(series):
    new_values = []
    index_list = []
    for items in series.iteritems():
        index = items[0]
        value = items[1]
        while value < 0 or value > 2 * np.pi:
            if value < 0:
                value = value + (2 * np.pi)
            elif value > 2 * np.pi:
                value = value - (2 * np.pi)
        new_values.append(value)
        index_list.append(index)
    new_series = pd.Series(new_values, index=index_list)
    return new_series

def iteration(Mk, e):
    Ek_values = []
    index_list = []

    for i in range(len(Mk.values)):
        Ek = 1
        Ek1 = Mk.values[i] + e.values[i] * np.sin(Ek)
        while abs(Ek - Ek1) > pow(10, -14):
            Ek = Ek1
            Ek1 = Mk.values[i] + e.values[i] * np.sin(Ek)
        Ek_values.append(Ek1)

    for items in Mk.iteritems():
        index = items[0]
        index_list.append(index)

    new_series = pd.Series(Ek_values, index=index_list)
    return new_series

def getNavigationData(system="GPS"):
    """Funkcja przetwarza dane z pliku nawigacyjnego na DataFrame, żeby użyć go potem w obliczaniu pozycji satelity"""

    """lista z satelitami z pliku nawigacyjnego i obserwacyjnego"""

    if system == "GPS":
        file = latest_file("O")
        obs = gr.load((str(file)), use='G')
        obs_satelites = list(obs['sv'].values)

        nav_file = latest_file("N", "GPS")
        nav = gr.load((str(nav_file)))

    elif system == "Galileo":
        file = latest_file("O")
        obs = gr.load((str(file)), use='E')
        obs_satelites = list(obs['sv'].values)

        nav_file = latest_file("N", "Galileo")
        nav = gr.load((str(nav_file)))

    """df z danymi, z pliku nawigacyjnego"""
    navigation_df = nav.to_dataframe()

    # wybieranie tylko wierszy dotyczących satelitów, występujących w pliku obserwacyjnym, a następnie usuwanie pustych wierszy"
    df_for_satellites = navigation_df.loc[obs_satelites]
    df_for_satellites = df_for_satellites.dropna(how='all')

    return df_for_satellites

def getSatECEF(system="GPS"):
    """Funkcja zwraca DataFrame z pozycją satelit w układzie ECEF"""

    file = latest_file("O")
    if system == "GPS":
        obs = gr.load((str(file)), use='G')
        times = getTimes(obs)

        # wczytywanie danych z pliku nawigacyjnego wykorzystując funkcję getNavigationData
        navigation_df = getNavigationData(system)

    elif system == "Galileo":
        obs = gr.load((str(file)), use='E')
        times = getTimes(obs)

        # wczytywanie danych z pliku nawigacyjnego wykorzystując funkcję getNavigationData
        navigation_df = getNavigationData(system)

    # tworzenie wektora z czasem dla epok z pliku obserwacyjnego,
    # a następnie macierze dla czasów z pliku nawigacyjnego i obsrwacyjnego
    nav_times = np.asarray(navigation_df.index.get_level_values(1), dtype='datetime64[ns]')
    obs_times = np.asarray(times, dtype='datetime64[ns]')

    # warunek na znajdowanie odpowiedniej epoki z pliku nawigacyjnego,
    # możliwie najbliższa, ale wcześniejsza niż z pliku obserwacyjnego
    # besttime = np.array([np.argmin(abs(nav_times - t)) for t in obs_times])

    df_for_times = navigation_df.reset_index(level=1).groupby(level=0).apply(lambda x: x.iloc[-1])

    # indeksowanie czasowe
    times_idx = np.vstack([obs_times] * int(len(df_for_times['time'])))
    times_idx = list(itertools.chain.from_iterable(times_idx))

    # tworzenie df o rozmiarze liczba satelitów x liczba epok w pliku obserwacyjnym
    df_to_compute = pd.DataFrame(np.repeat(df_for_times.values, len(obs_times), axis=0))

    # dodawanie kolejnych kolumny gpstimes
    df_to_compute.columns = df_for_times.columns
    df_to_compute['time'] = times_idx

    gpstime = np.array([getGpsTime(t) for t in times])
    gpstime = np.vstack([gpstime] * int(len(df_for_times['time'])))
    gpstime = list(itertools.chain.from_iterable(gpstime))

    # df przygotowany do obliczeń
    df_to_compute = df_to_compute.set_index('time')
    df_to_compute = df_to_compute.astype(float)
    df_to_compute['gpstime'] = gpstime

    sv_list = pd.Series(df_for_times.index.get_level_values(level=0))
    sv_list = sv_list.repeat(len(obs_times))
    df_to_compute['sv'] = sv_list.values

    # stałe
    GM = 3986005.0E8  # universal gravational constant
    OeDOT = 7.2921151467E-5  # WGS-84 value of the Earth's rotation rate

    # obliczenia
    tk = df_to_compute['gpstime'] - df_to_compute['Toe']  # epoka odniesienia efemryd
    a = pow(df_to_compute['sqrtA'], 2)  # dużą półoś orbity satelity
    n0 = np.sqrt(GM / pow(a, 3))  # ruch średni satelity
    n = n0 + df_to_compute['DeltaN']  # poprawiony ruch satelity
    Mk = df_to_compute['M0'] + n * tk  # anomalia średnia w epoce tk
    Mk = condition(Mk)
    Ek = iteration(Mk, df_to_compute['Eccentricity'])  # anomalia mimośrodkowa
    Ek = condition(Ek)
    v = np.arctan2((np.sqrt(1 - df_to_compute['Eccentricity'] ** 2) * np.sin(Ek)),
                   (np.cos(Ek) - df_to_compute['Eccentricity']))  # anomalia prawdziwa
    u = df_to_compute['omega'] + v  # argument szerokości
    u = condition(u)
    duk = df_to_compute['Cus'] * np.sin(2 * u) + df_to_compute['Cuc'] * np.cos(
        2 * u)  # poprawka dla argumentu szerokości
    drk = df_to_compute['Crs'] * np.sin(2 * u) + df_to_compute['Crc'] * np.cos(
        2 * u)  # poprawka dla promienia wodzącego
    dik = df_to_compute['Cis'] * np.sin(2 * u) + df_to_compute['Cic'] * np.cos(2 * u) + df_to_compute[
        'IDOT'] * tk  # poprawka dla kąta nachylenia orbity
    uk = u + duk  # poprawiony argument szerokości
    rk = a * (1 - df_to_compute['Eccentricity'] * np.cos(Ek)) + drk  # poprawiony promień wodzący
    ik = df_to_compute['Io'] + dik  # poprawiona wartość kąta nachylenia orbity
    omk = df_to_compute['Omega0'] + (df_to_compute['OmegaDot'] - OeDOT) * tk - (
            OeDOT * df_to_compute['Toe'])  # poprawiona długość węzła wstępującego orbityz
    omk = condition(omk)
    xi = rk * np.cos(uk)  # współrzędne satelity w płaszczyźnie orbity
    eta = rk * np.sin(uk)
    X = xi * np.cos(omk) - eta * np.cos(ik) * np.sin(omk)  # współrzędne geocentryczne satelity
    Y = xi * np.sin(omk) + eta * np.cos(ik) * np.cos(omk)
    Z = eta * np.sin(ik)

    satECEF = np.array([X, Y, Z]).T
    df_satECEF = pd.DataFrame(data=satECEF, index=sv_list.values, columns=['X', 'Y', 'Z'])
    df_satECEF['time'] = df_to_compute.index.to_list()
    df_satECEF['time_only'] = [x.strftime('%H:%M:%S') for x in df_satECEF['time'].tolist()]
    return df_satECEF

def getElevation_Azimuth_Angle(system="GPS"):
    """Ionospheric Pierce Point obliczone zgodnie z artykułem "Study of equatorial plasma bubbles
    using all sky imager and scintillation technique from Kolhapur station: a case study"""
    file = latest_file("O")
    obs = gr.load(str(file))

    if system == "GPS":
        SatECEF = getSatECEF("GPS")
    elif system == "Galileo":
        SatECEF = getSatECEF("Galileo")

    receiver_pos = obs.position
    rec_lat, rec_lon, rec_alt = ecef2geodetic(receiver_pos[0], receiver_pos[1], receiver_pos[2])

    # obliczenia azymutu i kąta elewacji
    az, el, r = ecef2aer(SatECEF['X'], SatECEF['Y'], SatECEF['Z'], rec_lat, rec_lon, rec_alt)

    SatECEF['sv'] = SatECEF.index
    SatECEF['Azimuth_deg'] = az
    SatECEF['Azimuth'] = np.radians(az)
    SatECEF['Elevation_deg'] = el
    SatECEF['Elevation'] = np.radians(el)

    # obliczenia IPP
    ipp_alt = 350
    R = 6371
    OMpp = np.pi / 2 - np.radians(SatECEF['Elevation_deg']) - np.arcsin(
        (R / (R + ipp_alt)) * np.cos(np.radians(SatECEF['Elevation_deg'])))
    ipp_lat = np.arcsin(
        np.sin(np.radians(rec_lat)) * np.cos(OMpp) + np.cos(np.radians(rec_lat)) * np.sin(OMpp) * np.cos(
            np.radians(az)))
    ipp_lon = np.radians(rec_lon) + np.arcsin(
        (np.sin(OMpp) * np.sin(np.radians(SatECEF['Azimuth_deg']))) / np.cos(ipp_lat))

    SatECEF['IPP_phi'] = np.degrees(ipp_lat)
    SatECEF['IPP_lambda'] = np.degrees(ipp_lon)

    # obliczenia sTEC
    sTEC_value = getRangeSlantTEC(system)
    SatECEF = SatECEF.reset_index()

    # dodawanie kolumn z df, z danymi o wartości sTEC
    # Complete_df = pd.concat([SatECEF, sTEC_value], axis=1)
    Complete_df = SatECEF.merge(sTEC_value, how='inner', left_on=["sv", "time_only"], right_on=["sv", "time_only"])

    Complete_df = Complete_df[Complete_df['sTEC'].notna()]

    Complete_df = Complete_df.set_index('index')

    # obliczenia vTEC
    Complete_df['vTEC'] = Complete_df['sTEC'] * np.sin(np.radians(Complete_df['Elevation_deg']))

    Complete_df = Complete_df.reset_index()
    return Complete_df
