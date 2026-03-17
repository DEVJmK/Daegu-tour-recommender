"""
네이버 지도 맛집 크롤러
- 관광지 주변 음식점 정보 수집 (가맹점명, 별점, 리뷰, 주소, 위치 등)
- 수집 결과: 맛집.csv

실행 전 준비사항:
  1. Chrome 브라우저 설치
  2. ChromeDriver 경로 설정 (WEBDRIVER_PATH 수정)
  3. pip install selenium webdriver-manager pandas
"""

import time
import pandas as pd
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

WEBDRIVER_PATH = "chromedriver.exe"   # 로컬 경로에 맞게 수정 (webdriver-manager 사용 시 생략 가능)
SEARCH_URL     = "https://map.naver.com/p?c=15.00,0,0,0,dh"
MAX_PLACES     = 35   # 관광지당 최대 수집 업체 수

# 수집할 관광지 목록
SEARCH_TERMS = [
    '강정고령보', '고산골공룡공원', '구암서원', '국립대구과학관', '국립대구박물관',
    '김광석다시그리기길', '두류공원', '달성공원', '달성토성공원', '달성습지',
    '대구미술관', '대구삼성라이온즈파크', '대구수목원', '대구스타디움',
    '대구평화시장 닭똥집 골목', '도동서원', '동성로', '동촌유원지',
    '동화사', '동화사집단시설지구', '팔공산케이블카', '들안길먹거리타운',
    '리조트스파벨리', '반고개무침회골목', '불로동', '고분공원',
    '비슬산자연휴양림', '사문진주막촌', '화원유원지', '서문시장', '서문야시장',
    '수성못', '신세계백화점대구점', '대구아쿠아리움', '안지랑곱창골목',
    '앞산전망대', '앞산케이블카', '약령시', '약령시한의약박물관',
    '옥연지', '송해공원', '옻골마을', '월광수변공원', '이월드',
    '칠성시장', '칠성야시장', '팔공산갓바위', '하중도',
]


# ─────────────────────────────────────────
# 드라이버 초기화
# ─────────────────────────────────────────

def init_driver():
    """Chrome WebDriver 초기화"""
    try:
        service = Service(executable_path=WEBDRIVER_PATH)
    except Exception:
        service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(SEARCH_URL)
    return driver


# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────

def switch_to_main_iframe(driver, wait):
    """메인 iframe 전환"""
    driver.switch_to.default_content()
    try:
        iframe = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe#main_iframe")))
        driver.switch_to.frame(iframe)
        return True
    except Exception:
        return False


def switch_to_search_iframe(driver, wait):
    """검색 결과 iframe 전환"""
    try:
        iframe = wait.until(EC.presence_of_element_located((By.ID, "searchIframe")))
        driver.switch_to.frame(iframe)
        return True
    except Exception as e:
        print("searchIframe 전환 실패:", e)
        return False


def input_search_query(driver, wait, place_name):
    """검색창에 장소명 입력 후 검색"""
    try:
        driver.switch_to.default_content()
        search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.input_search")))
        search_box.clear()
        search_box.send_keys(place_name)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)
        return True
    except Exception:
        print(f"{place_name} - 검색창을 찾을 수 없습니다.")
        return False


def clear_search_history(driver, wait):
    """검색 기록 초기화"""
    try:
        btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn_clear")))
        btn.click()
        time.sleep(1)
    except Exception:
        driver.refresh()
        time.sleep(2)


# ─────────────────────────────────────────
# 세부 정보 수집
# ─────────────────────────────────────────

def extract_place_info(driver, wait, term):
    """관광지 주변 음식점 최대 MAX_PLACES개 정보 수집"""
    place_info_list = []

    if not switch_to_search_iframe(driver, wait):
        return place_info_list

    containers = driver.find_elements(By.CLASS_NAME, "CHC5F")[:MAX_PLACES]

    for i, container in enumerate(containers):
        place_info = {"관광지": term}

        # 가맹점명
        try:
            place_info["가맹점명"] = container.find_element(By.CLASS_NAME, "TYaxT").text
        except NoSuchElementException:
            place_info["가맹점명"] = "N/A"

        # 상세 페이지 진입
        try:
            detail_link = container.find_element(By.CLASS_NAME, "place_bluelink.N_KDL")
            ActionChains(driver).move_to_element(detail_link).click().perform()
            time.sleep(2)

            driver.switch_to.default_content()
            detail_iframe = wait.until(EC.presence_of_element_located((By.ID, "entryIframe")))
            driver.switch_to.frame(detail_iframe)

            # 이미지
            try:
                place_info["가게 이미지 URL"] = driver.find_element(By.CSS_SELECTOR, "img.K0PDV").get_attribute("src")
            except Exception:
                place_info["가게 이미지 URL"] = "N/A"

            # 별점 / 리뷰 수
            try:
                review_counts = driver.find_elements(By.CSS_SELECTOR, "span.PXMot")
                if len(review_counts) == 3:
                    place_info["별점"]         = review_counts[0].text
                    place_info["방문자 리뷰 수"] = review_counts[1].text
                    place_info["블로그 리뷰 수"] = review_counts[2].text
                elif len(review_counts) == 2:
                    place_info["별점"]         = "N/A"
                    place_info["방문자 리뷰 수"] = review_counts[0].text
                    place_info["블로그 리뷰 수"] = review_counts[1].text
                else:
                    place_info["별점"] = place_info["방문자 리뷰 수"] = place_info["블로그 리뷰 수"] = "N/A"
            except Exception:
                place_info["별점"] = place_info["방문자 리뷰 수"] = place_info["블로그 리뷰 수"] = "N/A"

            # 주소
            try:
                place_info["주소"] = driver.find_element(By.CLASS_NAME, "LDgIH").text
            except NoSuchElementException:
                place_info["주소"] = "N/A"

            # 전화번호
            try:
                place_info["전화번호"] = driver.find_element(By.CLASS_NAME, "xlx7Q").text
            except Exception:
                place_info["전화번호"] = "N/A"

            # 웹사이트
            try:
                place_info["웹사이트 URL"] = driver.find_element(
                    By.CSS_SELECTOR, "a.place_bluelink.CHmqa"
                ).get_attribute("href")
            except Exception:
                place_info["웹사이트 URL"] = "N/A"

            # 방문자 리뷰 (최대 3개)
            try:
                reviews = driver.find_elements(By.CSS_SELECTOR, "a.pui__xtsQN-")
                place_info["방문자 리뷰"] = [r.text for r in reviews[:3]] if reviews else []
            except Exception:
                place_info["방문자 리뷰"] = []

            # 블로그 리뷰
            try:
                blog_titles   = driver.find_elements(By.CSS_SELECTOR, "div.pui__dGLDWy")
                blog_reviews  = driver.find_elements(By.CSS_SELECTOR, "span.pui__xtsQN-")
                place_info["블로그 제목"] = [t.text for t in blog_titles]  if blog_titles  else []
                place_info["블로그 리뷰"] = [r.text for r in blog_reviews] if blog_reviews else []
            except Exception:
                place_info["블로그 제목"] = []
                place_info["블로그 리뷰"] = []

            # 위치 URL
            location_url = "N/A"
            driver.switch_to.default_content()
            try:
                btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "naver-splugin"))
                )
                ActionChains(driver).move_to_element(btn).click(btn).perform()
                time.sleep(2)
                url_el = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-button="copyurl"]'))
                )
                location_url = url_el.get_attribute("href")
            except Exception:
                pass
            place_info["위치값 주소"] = location_url

            # 검색 iframe 복귀
            driver.switch_to.default_content()
            switch_to_search_iframe(driver, wait)

        except Exception as e:
            print(f"  [{place_info.get('가맹점명', 'N/A')}] 상세 정보 수집 실패: {e}")

        place_info_list.append(place_info)
        print(f"  수집 완료 [{i+1}]: {place_info.get('가맹점명', 'N/A')}")

    return place_info_list


# ─────────────────────────────────────────
# 좌표 추출 (위치 URL → 위도/경도)
# ─────────────────────────────────────────

def get_coordinates_from_url(driver, location_url):
    """네이버 지도 URL에서 위도/경도 파싱"""
    import re
    driver.get("https://xn--yq5bk9r.com/blog/map-coordinates")
    latitude, longitude = "N/A", "N/A"

    try:
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "default-input"))
        )
        input_box.clear()
        input_box.send_keys(location_url)

        convert_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "default-button"))
        )
        convert_btn.click()

        result_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[@class='default-button h-10']/following-sibling::div/div/div")
            )
        )
        text = result_div.text
        if "위도" in text and "경도" in text:
            latitude  = text.split("위도 : ")[1].split(",")[0].strip()
            longitude = text.split("경도 : ")[1].split("\n")[0].strip()
    except Exception as e:
        print(f"좌표 조회 실패 ({location_url}): {e}")

    return latitude, longitude


def add_coordinates(driver, data):
    """data 리스트에 위도/경도 추가"""
    for item in data:
        url = item.get("위치값 주소", "N/A")
        if url != "N/A":
            lat, lng = get_coordinates_from_url(driver, url)
        else:
            lat, lng = "N/A", "N/A"
        item["위도"] = lat
        item["경도"] = lng
        print(f"  좌표 추가: {item.get('가맹점명', 'N/A')} → ({lat}, {lng})")


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────

def main():
    driver = init_driver()
    wait   = WebDriverWait(driver, 10)
    data   = []

    for term in SEARCH_TERMS:
        print(f"\n[검색 중] {term}")

        # ① 관광지 검색
        input_search_query(driver, wait, term)
        clear_search_history(driver, wait)

        # ② 음식점 탭 검색
        input_search_query(driver, wait, "음식점")
        time.sleep(3)

        # ③ 정보 수집
        place_info = extract_place_info(driver, wait, term)
        data.extend(place_info)

        # ④ 페이지 새로고침 후 다음 검색어 준비
        driver.refresh()
        try:
            clear_btn = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "btn_clear")))
            clear_btn.click()
        except Exception:
            pass
        time.sleep(2)

    # ⑤ 좌표 추가
    add_coordinates(driver, data)
    driver.quit()

    # ⑥ CSV 저장
    df = pd.DataFrame(data)
    df.to_csv("data/맛집.csv", index=False, encoding="utf-8-sig")
    print("저장 완료")


if __name__ == "__main__":
    main()
