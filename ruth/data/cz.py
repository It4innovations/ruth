
import os
from enum import Enum

from ..metaclasses import Singleton
from .border import Border, BorderType


CZ0 = {"country": "Czech Republic"}

# ------------------------------------------------------------------------------
# NUTS 3


class County(Enum):
    CZ010 = {**CZ0, "county": "Hlavní město Praha"}
    CZ020 = {**CZ0, "county": "Středočeský kraj"}
    CZ031 = {**CZ0, "county": "Jihočeský kraj"}
    CZ032 = {**CZ0, "county": "Plzeňský kraj"}
    CZ041 = {**CZ0, "county": "Karlovarský kraj"}
    CZ042 = {**CZ0, "county": "Ústecký kraj"}
    CZ051 = {**CZ0, "county": "Liberecký kraj"}
    CZ052 = {**CZ0, "county": "Královehradecký kraj"}
    CZ053 = {**CZ0, "county": "Pardubický kraj"}
    CZ063 = {**CZ0, "county": "Kraj Vysočina"}
    CZ064 = {**CZ0, "county": "Jihomoravský kraj"}
    CZ071 = {**CZ0, "county": "Olomoucký kraj"}
    CZ072 = {**CZ0, "county": "Zlínský kraj"}
    CZ080 = {**CZ0, "county": "Moravskoslezský kraj"}


# ------------------------------------------------------------------------------
# LAU 1


class District_CZ020(Enum):  # Stredocesky kraj
    CZ0201 = "CZ0201"  # district: Benešov
    CZ0202 = "CZ0202"  # district: Beroun
    CZ0203 = "CZ0203"  # district: Kladno
    CZ0204 = "CZ0204"  # district: Kolín
    CZ0205 = "CZ0205"  # district: Kutná Hora
    CZ0206 = "CZ0206"  # district: Mělník
    CZ0207 = "CZ0207"  # district: Mladá Boleslav
    CZ0208 = "CZ0208"  # district: Nymburk
    CZ0209 = "CZ0209"  # district: Praha-východ
    CZ020A = "CZ020A"  # district: Praha-západ
    CZ020B = "CZ020B"  # district: Příbram
    CZ020C = "CZ020C"  # district: Rakovník


class District_CZ031(Enum):  # jihocesky kraj
    CZ0311 = "CZ0311"  # district: České Budějovice
    CZ0312 = "CZ0312"  # district: Český Krumlov
    CZ0313 = "CZ0313"  # district: Jindřichův Hradec
    CZ0314 = "CZ0314"  # district: Písek
    CZ0315 = "CZ0315"  # district: Prachatice
    CZ0316 = "CZ0316"  # district: Strakonice
    CZ0317 = "CZ0317"  # district: Tábor


class District_CZ032(Enum):  # Plzensky kraj
    CZ0321 = "CZ0321"  # district: Domažlice
    CZ0322 = "CZ0322"  # district: Klatovy
    CZ0323 = "CZ0323"  # district: Plzeň-město
    CZ0324 = "CZ0324"  # district: Plzeň-jih
    CZ0325 = "CZ0325"  # district: Plzeň-sever
    CZ0326 = "CZ0326"  # district: Rokycany
    CZ0327 = "CZ0327"  # district: Tachov


class District_CZ041(Enum):  # Karlovarsky kraj
    CZ0411 = "CZ0411"  # district: Cheb
    CZ0412 = "CZ0412"  # district: Karlovy Vary
    CZ0413 = "CZ0413"  # district: Sokolov


class District_CZ042(Enum):  # Ustecky kraj
    CZ0421 = "CZ0421"  # district: Děčín
    CZ0422 = "CZ0422"  # district: Chomutov
    CZ0423 = "CZ0423"  # district: Litoměřice
    CZ0424 = "CZ0424"  # district: Louny
    CZ0425 = "CZ0425"  # district: Most
    CZ0426 = "CZ0426"  # district: Teplice
    CZ0427 = "CZ0427"  # district: Ústí nad Labem


class District_CZ051(Enum):  # Liberecky kraj
    CZ0511 = "CZ0511"  # district: Česká Lípa
    CZ0512 = "CZ0512"  # district: Jablonec nad Nisou
    CZ0513 = "CZ0513"  # district: Liberec
    CZ0514 = "CZ0514"  # district: Semily


class District_CZ052(Enum):  # Kralovehradecky kraj
    CZ0521 = "CZ0521"  # district: Hradec Králové
    CZ0522 = "CZ0522"  # district: Jičín
    CZ0523 = "CZ0523"  # district: Náchod
    CZ0524 = "CZ0524"  # district: Rychnov nad Kněžnou
    CZ0525 = "CZ0525"  # district: Trutnov


class District_CZ053(Enum):  # Pardubicky kraj
    CZ0531 = "CZ0531"  # district: Chrudim
    CZ0532 = "CZ0532"  # district: Pardubice
    CZ0533 = "CZ0533"  # district: Svitavy
    CZ0534 = "CZ0534"  # district: Ústí nad Orlicí


class District_CZ063(Enum):  # Kraj Vysocina
    CZ0631 = "CZ0631"  # district: Havlíčkův Brod
    CZ0632 = "CZ0632"  # district: Jihlava
    CZ0633 = "CZ0633"  # district: Pelhřimov
    CZ0634 = "CZ0634"  # district: Třebíč
    CZ0635 = "CZ0635"  # district: Žďár nad Sázavou


class District_CZ064(Enum):  # Jihomoravsky kraj
    CZ0641 = "CZ0641"  # district: Blansko
    CZ0642 = "CZ0642"  # district: Brno-město
    CZ0643 = "CZ0643"  # district: Brno-venkov
    CZ0644 = "CZ0644"  # district: Břeclav
    CZ0645 = "CZ0645"  # district: Hodonín
    CZ0646 = "CZ0646"  # district: Vyškov
    CZ0647 = "CZ0647"  # district: Znojmo


class District_CZ071(Enum):  # olomoucky kraj
    CZ0711 = "CZ0711"  # district: Jeseník
    CZ0712 = "CZ0712"  # district: Olomouc
    CZ0713 = "CZ0713"  # district: Prostějov
    CZ0714 = "CZ0714"  # district: Přerov
    CZ0715 = "CZ0715"  # district: Šumperk


class District_CZ072(Enum):  # Zlinsky kraj
    CZ0721 = "CZ0721"  # district: Kroměříž
    CZ0722 = "CZ0722"  # district: Uherské Hradiště
    CZ0723 = "CZ0723"  # district: Vsetín
    CZ0724 = "CZ0724"  # district: Zlín


class District_CZ080(Enum):  # Moravskoslezky kraj
    CZ0801 = "CZ0801"  # district: Bruntál
    CZ0802 = "CZ0802"  # district: Frýdek-Místek
    CZ0803 = "CZ0803"  # district: Karviná
    CZ0804 = "CZ0804"  # district: Nový Jičín
    CZ0805 = "CZ0805"  # district: Opava
    CZ0806 = "CZ0806"  # district: Ostrava-město



class Boundary(metaclass=Singleton):
    def __init__(self, data_dir="./data", load_from_cache=True):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self._all = []
        # Czech republic boundary
        self.CZ0 = Border("CZ0", CZ0, BorderType.COUNTRY, data_dir, load_from_cache)
        self._all.append(self.CZ0)

        # Boundary of czech counties
        self._counties = self._load_boundaries(
            self.CZ0, County, BorderType.COUNTY, data_dir, load_from_cache
        )

        # Boundary of particular districts
        self._district_cz020 = self._load_boundaries(
            self.CZ020, District_CZ020, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz031 = self._load_boundaries(
            self.CZ031, District_CZ031, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz032 = self._load_boundaries(
            self.CZ032, District_CZ032, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz041 = self._load_boundaries(
            self.CZ041, District_CZ041, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz042 = self._load_boundaries(
            self.CZ042, District_CZ042, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz051 = self._load_boundaries(
            self.CZ051, District_CZ051, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz052 = self._load_boundaries(
            self.CZ052, District_CZ052, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz053 = self._load_boundaries(
            self.CZ053, District_CZ053, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz063 = self._load_boundaries(
            self.CZ063, District_CZ063, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz064 = self._load_boundaries(
            self.CZ064, District_CZ064, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz071 = self._load_boundaries(
            self.CZ071, District_CZ071, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz072 = self._load_boundaries(
            self.CZ072, District_CZ072, BorderType.DISTRICT, data_dir, load_from_cache
        )

        self._district_cz080 = self._load_boundaries(
            self.CZ080, District_CZ080, BorderType.DISTRICT, data_dir, load_from_cache
        )

    def _load_boundaries(self, parent, places, border_type, data_dir, load_from_cache):
        boundaries = []
        for p in places:
            b = Border(p.name, p.value, border_type, data_dir, load_from_cache)

            boundaries.append(b)
            setattr(self, p.name, b)

        if parent is not None:
            parent.add(boundaries)

        self._all.extend(boundaries)
        return boundaries

    def __iter__(self):
        return iter(self._all)
