# 필요한 라이브러리 임포트
import streamlit as st
from google.generativeai import GenerativeModel
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# ============================================================================
# 에이전틱 워크플로우 기반 인사(HR) 파트너 시스템
# 3명의 특화된 인사 전문가가 팀을 이루어 직장인을 지원
# ============================================================================

class HRPartnerTeam:
    """
    AI 기반 인사(HR) 파트너 팀을 관리하는 클래스
    각 전문가의 협업을 조율하고 최종 결과를 제공
    """
    
    def __init__(self, api_key):
        """
        인사 파트너 팀 초기화
        Args:
            api_key (str): Google AI API 키
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = GenerativeModel('gemini-2.5-pro-preview-05-06')
        
        # 3명의 특화된 인사 전문가 초기화
        self.hr_specialist = HRSpecialist(self.model)  # 인사 정책 및 제도 전문가
        self.career_partner = CareerPartner(self.model)    # 경력 개발 및 성장 전문가 (CareerCoach -> CareerPartner)
        self.workplace_advisor = WorkplaceAdvisor(self.model)  # 직장 문화 및 대인관계 전문가
        
        # 워크플로우 로그 초기화
        self.workflow_logs = []
    
    def get_hr_advice(self, service_type, input_data):
        """
        사용자 요청에 따라 3명의 전문가가 순차적으로 협업하여 조언 제공
        Args:
            service_type (str): 요청 서비스 유형
            input_data (dict): 사용자 입력 데이터
        Returns:
            dict: 각 전문가의 조언을 포함한 결과
        """
        # 워크플로우 기록 시작
        workflow_log = {
            "service_type": service_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experts_involved": ["HRSpecialist", "CareerPartner", "WorkplaceAdvisor"], # CareerCoach -> CareerPartner
            "steps": []
        }
        
        # 1단계: 인사 전문가의 초기 분석 및 정책/제도 설명
        st.markdown("### 1단계: 인사 정책 및 제도 분석 중...")
        with st.spinner("인사 전문가가 분석 중입니다..."):
            hr_analysis = self.hr_specialist.analyze(service_type, input_data)
            workflow_log["steps"].append({
                "expert": "HRSpecialist",
                "action": "hr_analysis"
            })
        
        # 2단계: 경력 파트너의 성장 및 개발 관점 추가
        st.markdown("### 2단계: 경력 개발 관점 분석 중...")
        with st.spinner("경력 파트너가 분석 중입니다..."): # 경력 코치 -> 경력 파트너
            career_analysis = self.career_partner.enhance(hr_analysis, service_type, input_data) # career_coach -> career_partner
            workflow_log["steps"].append({
                "expert": "CareerPartner", # CareerCoach -> CareerPartner
                "action": "career_enhancement"
            })
        
        # 3단계: 직장 어드바이저의 문화 및 대인관계 관점 최적화
        st.markdown("### 3단계: 직장 문화 및 대인관계 관점 분석 중...")
        with st.spinner("직장 어드바이저가 최종 조언을 준비 중입니다..."):
            workplace_advice = self.workplace_advisor.finalize(career_analysis, service_type, input_data)
            workflow_log["steps"].append({
                "expert": "WorkplaceAdvisor",
                "action": "workplace_finalization"
            })
        
        # 워크플로우 로그 저장
        self.workflow_logs.append(workflow_log)
        
        # 각 전문가별 결과를 모두 반환
        return {
            "hr": hr_analysis,
            "career": career_analysis,
            "workplace": workplace_advice
        }


class HRSpecialist:
    """
    인사 정책 및 제도 전문가
    노동법, 인사 제도, 복리후생, 평가 시스템 등에 대한 전문 지식 제공
    """
    
    def __init__(self, model):
        self.model = model
        self.expertise = "hr_policies"
        self.expert_name = "김민준 인사 전문가"  # 김정책 -> 김민준
        self.expert_intro = """
        안녕하세요, 김민준 인사 전문가입니다. 
        저는 인사 정책, 노동법, 복리후생, 평가 시스템 등 회사의 공식적인 인사 제도에 대한 전문 지식을 제공합니다.
        15년간의 대기업 인사팀 및 노무 컨설팅 경험을 바탕으로 정확하고 실용적인 인사 관련 정보를 안내해 드리겠습니다.
        """
    
    def analyze(self, service_type, input_data):
        """
        사용자 요청에 대한 인사 정책 및 제도 분석
        """
        # 서비스 유형별 맞춤 프롬프트 생성
        if service_type == "고용/계약":
            prompt = self._create_employment_prompt(input_data)
        elif service_type == "급여/복리후생":
            prompt = self._create_compensation_prompt(input_data)
        elif service_type == "평가/성과":
            prompt = self._create_performance_prompt(input_data)
        elif service_type == "직장 내 문제":
            prompt = self._create_workplace_issue_prompt(input_data)
        elif service_type == "경력 개발":
            prompt = self._create_career_hr_prompt(input_data)
        else:
            prompt = self._create_general_hr_prompt(input_data, service_type)
        
        # 전문가 정보 추가
        prompt = f"""
        당신은 '{self.expert_name}'이라는 인사 정책 및 제도 전문가입니다.
        {self.expert_intro}
        
        {prompt}
        
        분석 결과에는 관련 인사 정책, 제도, 법규, 일반적인 기업 관행을 반드시 포함해 주세요.
        정확한 정보를 제공하되, 이해하기 쉬운 언어로 설명해 주세요.
        법적 조언이 필요한 경우 전문가 상담을 권고하는 문구를 포함해 주세요.
        """
        
        # AI 모델을 통한 응답 생성
        response = self.model.generate_content(prompt)
        return response.text
    
    def _create_employment_prompt(self, input_data):
        return f"""
        다음 고용/계약 관련 질문에 대해 인사 정책 관점에서 분석해주세요:
        
        {input_data.get('question', '')}
        
        다음 항목을 포함하는 분석을 제공해주세요:
        1. 관련 노동법 및 법적 권리/의무
        2. 일반적인 고용 계약 관행 및 조건
        3. 회사별 차이가 있을 수 있는 정책 영역
        4. 근로자가 확인하거나 협상할 수 있는 사항
        5. 계약/고용 과정에서 주의해야 할 점
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        근무 산업: {input_data.get('industry', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        """
    
    def _create_compensation_prompt(self, input_data):
        return f"""
        다음 급여/복리후생 관련 질문에 대해 인사 정책 관점에서 분석해주세요:
        
        {input_data.get('question', '')}
        
        다음 항목을 포함하는 분석을 제공해주세요:
        1. 관련 법적 기준 및 의무 사항
        2. 일반적인 업계 보상 체계 및 관행
        3. 법정/비법정 복리후생 구분
        4. 세금 및 사회보험 관련 고려사항
        5. 협상 가능한 영역 및 접근 방법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        근무 산업: {input_data.get('industry', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        지역: {input_data.get('location', '정보 없음')}
        """
    
    def _create_performance_prompt(self, input_data):
        return f"""
        다음 평가/성과 관련 질문에 대해 인사 정책 관점에서 분석해주세요:
        
        {input_data.get('question', '')}
        
        다음 항목을 포함하는 분석을 제공해주세요:
        1. 일반적인 기업 평가 시스템 구조
        2. 평가 결과의 활용 (승진, 보상, 교육 연계)
        3. 객관적 평가를 위한 제도적 장치
        4. 평가 관련 근로자의 권리와 이의제기 절차
        5. 평가 시스템 유형별 특징과 대응 방법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        근무 산업: {input_data.get('industry', '정보 없음')}
        회사 규모: {input_data.get('company_size', '정보 없음')}
        """
    
    def _create_workplace_issue_prompt(self, input_data):
        return f"""
        다음 직장 내 문제 관련 질문에 대해 인사 정책 관점에서 분석해주세요:
        
        {input_data.get('question', '')}
        
        다음 항목을 포함하는 분석을 제공해주세요:
        1. 관련 법규 및 보호 장치
        2. 회사 내 공식적 해결 절차 및 채널
        3. 인사팀/고충처리위원회의 역할
        4. 문제 해결을 위한 공식 문서화 방법
        5. 외부 지원/상담 기관 정보
        
        문제 유형: {input_data.get('issue_type', '정보 없음')}
        직급/직책: {input_data.get('position', '정보 없음')}
        회사 규모: {input_data.get('company_size', '정보 없음')}
        """
    
    def _create_career_hr_prompt(self, input_data):
        return f"""
        다음 경력 개발 관련 질문에 대해 인사 정책 관점에서 분석해주세요:
        
        {input_data.get('question', '')}
        
        다음 항목을 포함하는 분석을 제공해주세요:
        1. 기업의 일반적인 경력 개발 제도
        2. 직급/승진 체계 및 요건
        3. 사내 교육/훈련 프로그램 활용
        4. 자기계발 지원 제도 및 신청 방법
        5. 경력 개발 계획과 인사평가 연계 방법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        목표 경력 경로: {input_data.get('career_goal', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        """
    
    def _create_general_hr_prompt(self, input_data, service_type):
        return f"""
        다음 {service_type} 관련 질문에 대해 인사 정책 관점에서 분석해주세요:
        
        {input_data.get('question', '')}
        
        관련 인사 정책, 법규, 제도, 일반적 관행을 포함한 분석을 제공해주세요.
        정확한 정보를 바탕으로 실용적인 조언을 제공해 주세요.
        """


class CareerPartner:
    """
    경력 개발 및 성장 전문가
    경력 경로, 역량 개발, 전문성 향상, 이직/승진 전략 등에 대한 조언 제공
    """
    
    def __init__(self, model):
        self.model = model
        self.expertise = "career_development"
        self.expert_name = "이서연 경력 파트너"
        self.expert_intro = """
        안녕하세요, 이서연 경력 파트너입니다.
        저는 직장인의 경력 개발, 역량 향상, 승진/이직 전략 등 전문적 성장을 지원합니다.
        12년간의 경력 코칭 및 인재 개발 경험을 통해 여러분의 커리어 여정을 함께 설계하겠습니다.
        """
    
    def enhance(self, previous_analysis, service_type, input_data):
        """
        인사 전문가의 분석을 바탕으로 경력 개발 관점의 조언 추가
        """
        # 서비스 유형별 맞춤 프롬프트 생성
        if service_type == "고용/계약":
            prompt = self._create_employment_career_prompt(previous_analysis, input_data)
        elif service_type == "급여/복리후생":
            prompt = self._create_compensation_career_prompt(previous_analysis, input_data)
        elif service_type == "평가/성과":
            prompt = self._create_performance_career_prompt(previous_analysis, input_data)
        elif service_type == "직장 내 문제":
            prompt = self._create_workplace_issue_career_prompt(previous_analysis, input_data)
        elif service_type == "경력 개발":
            prompt = self._create_career_development_prompt(previous_analysis, input_data)
        else:
            prompt = self._create_general_career_prompt(previous_analysis, input_data, service_type)
        
        # 전문가 정보 추가
        prompt = f"""
        당신은 '{self.expert_name}'이라는 경력 개발 전문 파트너입니다.
        {self.expert_intro}
        
        인사 전문가가 제공한 다음 분석을 검토하고, 경력 개발 관점에서 보완해주세요:
        
        === 인사 전문가의 분석 ===
        {previous_analysis}
        === 분석 끝 ===
        
        {prompt}
        
        실질적인 경력 성장 전략, 역량 개발 방법, 전문성 향상 방안을 반드시 포함해 주세요.
        실현 가능하고 구체적인 조언을 제공하며, 업계 트렌드와 실제 현장의 경험을 반영해 주세요.
        """
        
        # AI 모델을 통한 응답 생성
        response = self.model.generate_content(prompt)
        return response.text
    
    def _create_employment_career_prompt(self, previous_analysis, input_data):
        return f"""
        고용/계약 관련 상황에서의 경력 개발 관점 조언을 제공해주세요:
        
        1. 계약 조건과 경력 발전 기회 연계 방법
           - 역량 개발 지원 조항 확인/협상 포인트
           - 경력 성장에 유리한 계약 조건
           - 성장 가능성 평가를 위한 질문/체크리스트
        
        2. 계약 유형별 경력 개발 전략
           - 계약 형태에 따른 경력 관리 접근법
           - 단기/장기 계약별 역량 개발 우선순위
           - 계약 기간 내 최대 성장을 위한 전략
        
        3. 입사 초기/계약 갱신 시 경력 개발 기회 확보
           - OJT/멘토링 프로그램 활용법
           - 성장 기회 협상 및 요청 방법
           - 경력 개발 계획 수립 및 공유 방법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        경력 목표: {input_data.get('career_goal', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        """
    
    def _create_compensation_career_prompt(self, previous_analysis, input_data):
        return f"""
        급여/복리후생 관련 상황에서의 경력 개발 관점 조언을 제공해주세요:
        
        1. 보상 패키지와 경력 개발 연계
           - 교육/자기계발 지원 제도 활용법
           - 성과급/인센티브 시스템을 성장 동력으로 활용
           - 복리후생을 역량 개발에 활용하는 방법
        
        2. 급여 협상과 경력 가치 증대
           - 역량/경험에 따른 시장 가치 평가
           - 급여 협상 시 역량/성과 활용 전략
           - 금전적/비금전적 보상 균형 설계
        
        3. 보상 체계 이해를 통한 경력 계획
           - 회사/업계 보상 체계 파악 방법
           - 경력 단계별 적정 보상 범위
           - 핵심 역량 개발을 통한 가치 증대 방법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        급여 정보: {input_data.get('salary', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        """
    
    def _create_performance_career_prompt(self, previous_analysis, input_data):
        return f"""
        평가/성과 관련 상황에서의 경력 개발 관점 조언을 제공해주세요:
        
        1. 평가 시스템을 경력 개발에 활용하는 방법
           - 평가 결과 해석 및 역량 갭 분석
           - 피드백을 성장 기회로 전환하는 접근법
           - 목표 설정을 통한 경력 방향성 명확화
        
        2. 성과 향상을 위한 역량 개발 전략
           - 직무별 핵심 역량 식별 및 집중 개발
           - 평가 항목별 개선 계획 수립
           - 상사/동료 피드백 활용 방법
        
        3. 평가 과정을 통한 경력 계획 조정
           - 강점/약점 파악을 통한 경력 방향 설정
           - 평가 면담을 성장 기회로 활용하는 방법
           - 역량 진단에 기반한 교육/훈련 계획
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        평가 고민: {input_data.get('performance_concern', '정보 없음')}
        경력 목표: {input_data.get('career_goal', '정보 없음')}
        """
    
    def _create_workplace_issue_career_prompt(self, previous_analysis, input_data):
        return f"""
        직장 내 문제 상황에서의 경력 개발 관점 조언을 제공해주세요:
        
        1. 문제 상황을 성장 기회로 전환하는 접근법
           - 갈등/도전 상황에서의 역량 개발 가능성
           - 문제 해결 과정을 통한 리더십 발휘 기회
           - 어려운 상황을 통한 회복탄력성 구축
        
        2. 문제 해결과 전문성/경력 이미지 관리
           - 문제 상황에서의 전문가적 태도 유지
           - 건설적 해결 과정을 통한 평판 관리
           - 상황 해결 후 회복 및 성장 계획
        
        3. 장기적 경력 관점에서의 대응 전략
           - 현 상황이 경력에 미치는 영향 평가
           - 필요시 대안적 경력 경로 탐색
           - 문제 경험을 미래 직무에 활용하는 방법
        
        문제 유형: {input_data.get('issue_type', '정보 없음')}
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        """
    
    def _create_career_development_prompt(self, previous_analysis, input_data):
        return f"""
        경력 개발에 대한 종합적인 조언을 제공해주세요:
        
        1. 개인 맞춤형 경력 개발 로드맵
           - 현재 위치 진단 및 목표 설정 방법
           - 단계별 성장 계획 및 이정표
           - 역량 개발 우선순위 및 접근법
        
        2. 역량 향상을 위한 실천적 전략
           - 직무/산업별 핵심 역량 개발 방법
           - 공식/비공식 학습 기회 활용
           - 자기주도 학습 및 실천 계획
        
        3. 경력 성장을 위한 전략적 포지셔닝
           - 조직 내 가시성 확보 및 영향력 구축
           - 네트워킹 및 멘토십 활용 방법
           - 승진/이직을 위한 포트폴리오 구축
        
        4. 미래 트렌드에 대비한 역량 개발
           - 산업/직무 변화 예측 및 대응
           - 기술적/소프트 스킬 균형 개발
           - 지속적 성장을 위한 학습 습관
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        경력 목표: {input_data.get('career_goal', '정보 없음')}
        관심 역량 영역: {input_data.get('skill_interests', '정보 없음')}
        경력 기간: {input_data.get('experience', '정보 없음')}
        """
    
    def _create_general_career_prompt(self, previous_analysis, input_data, service_type):
        return f"""
        다음 {service_type} 관련 질문에 대해 경력 개발 관점에서 분석해주세요:
        
        질문:
        {input_data.get('question', '')}
        
        경력 성장 전략, 역량 개발 방법, 직무 전문성 향상 방안을 구체적으로 제시해주세요.
        """


class WorkplaceAdvisor:
    """
    직장 문화 및 대인관계 전문가
    직장 내 소통, 갈등 관리, 팀워크, 리더십, 직장 문화 적응 등에 대한 조언 제공
    """
    
    def __init__(self, model):
        self.model = model
        self.expertise = "workplace_relations"
        self.expert_name = "박지훈 직장 어드바이저"
        self.expert_intro = """
        안녕하세요, 박지훈 직장 어드바이저입니다.
        저는 직장 내 대인관계, 소통 전략, 갈등 관리, 조직 문화 적응 등 직장 생활의 인간적 측면을 전문으로 합니다.
        13년간의 조직심리 컨설팅 및 기업 문화 연구 경험을 통해 건강하고 생산적인 직장 생활을 위한 실질적 조언을 제공하겠습니다.
        """
    
    def finalize(self, previous_analysis, service_type, input_data):
        """
        인사 전문가와 경력 파트너의 분석을 바탕으로 직장 문화 및 대인관계 관점의 최종 조언 제공
        """
        # 서비스 유형별 맞춤 프롬프트 생성
        if service_type == "고용/계약":
            prompt = self._create_employment_workplace_prompt(previous_analysis, input_data)
        elif service_type == "급여/복리후생":
            prompt = self._create_compensation_workplace_prompt(previous_analysis, input_data)
        elif service_type == "평가/성과":
            prompt = self._create_performance_workplace_prompt(previous_analysis, input_data)
        elif service_type == "직장 내 문제":
            prompt = self._create_workplace_issue_complete_prompt(previous_analysis, input_data)
        elif service_type == "경력 개발":
            prompt = self._create_career_workplace_prompt(previous_analysis, input_data)
        else:
            prompt = self._create_general_workplace_prompt(previous_analysis, input_data, service_type)
        
        # 전문가 정보 추가
        prompt = f"""
        당신은 '{self.expert_name}'이라는 직장 문화 및 대인관계 전문가입니다.
        {self.expert_intro}
        
        인사 전문가와 경력 파트너가 제공한, 다음 분석을 검토하고 최종적으로 완성해주세요:
        
        === 이전 전문가들의 분석 ===
        {previous_analysis}
        === 분석 끝 ===
        
        {prompt}
        
        최종 조언에는 다음 세 전문가의 관점이 균형있게 통합되어야 합니다:
        1. 인사 전문가 (정책/제도 관점)
        2. 경력 파트너 (성장/개발 관점)
        3. 직장 어드바이저 (문화/관계 관점)
        
        실용적이고 적용 가능한 조언, 건강한 직장 생활을 위한 대인관계 전략, 조직 문화 적응 방법을 포함해 주세요.
        """
        
        # AI 모델을 통한 응답 생성
        response = self.model.generate_content(prompt)
        return response.text
    
    def _create_employment_workplace_prompt(self, previous_analysis, input_data):
        return f"""
        고용/계약 관련 상황에서의 직장 문화 및 대인관계 관점 조언을 제공해주세요:
        
        1. 고용/계약 과정에서의 문화적 요소 파악
           - 면접/협상 과정에서 조직 문화 파악법
           - 계약 조건에 반영된 조직 가치 읽어내기
           - 공식/비공식 문화 간 차이점 식별
        
        2. 입사 초기 조직 적응 및 관계 형성
           - 효과적인 온보딩을 위한 소통 전략
           - 핵심 관계자와의 라포 형성 방법
           - 조직 내 성공적 첫인상 구축
        
        3. 계약 관계에서의 심리적 계약 관리
           - 명시적/암묵적 기대 사항 파악
           - 건강한 경계 설정 및 유지
           - 조직 내 지위/역할에 맞는 관계 형성
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        직장 문화 유형: {input_data.get('workplace_culture', '정보 없음')}
        개인 성향: {input_data.get('personality', '정보 없음')}
        """
    
    def _create_compensation_workplace_prompt(self, previous_analysis, input_data):
        return f"""
        급여/복리후생 관련 상황에서의 직장 문화 및 대인관계 관점 조언을 제공해주세요:
        
        1. 보상 관련 대화의 문화적 접근
           - 급여/복리후생 논의를 위한 적절한 소통 방식
           - 문화별 보상 협상 접근법 차이
           - 예민한 주제 다루기 위한 대화 전략
        
        2. 보상 시스템과 팀 역학 관계
           - 보상 차이가 팀 관계에 미치는 영향 관리
           - 공정성 인식과 관계 유지 전략
           - 보상 관련 갈등 예방 및 해소법
        
        3. 복리후생 활용과 직장 문화 참여
           - 복리후생을 통한 조직 네트워킹 기회
           - 복리후생 활용 시 문화적 고려사항
           - 팀/부서별 비공식 혜택 파악 및 활용
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        조직 규모: {input_data.get('company_size', '정보 없음')}
        문화적 특성: {input_data.get('workplace_culture', '정보 없음')}
        """
    
    def _create_performance_workplace_prompt(self, previous_analysis, input_data):
        return f"""
        평가/성과 관련 상황에서의 직장 문화 및 대인관계 관점 조언을 제공해주세요:
        
        1. 평가 과정에서의 효과적 소통 전략
           - 평가자/피평가자 관계 관리 방법
           - 건설적 피드백 주고받는 대화 기술
           - 어려운 평가 결과 수용 및 대응법
        
        2. 성과 향상을 위한 관계적 접근
           - 멘토/롤모델 식별 및 관계 구축
           - 동료 피드백 요청 및 활용 방법
           - 상사와의 기대치 조율 기술
        
        3. 평가 문화에 따른 적응 전략
           - 경쟁/협력 중심 평가 문화별 접근법
           - 공식/비공식 평가 요소 파악
           - 평가 시즌의 조직 분위기 관리법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        팀 구조: {input_data.get('team_structure', '정보 없음')}
        평가 문화: {input_data.get('evaluation_culture', '정보 없음')}
        """
    
    def _create_workplace_issue_complete_prompt(self, previous_analysis, input_data):
        return f"""
        직장 내 문제 상황에서의 종합적인 문화 및 대인관계 조언을 제공해주세요:
        
        1. 문제 상황의 인간관계 역학 분석
           - 갈등 당사자 및 이해관계자 매핑
           - 공식/비공식 권력 구조 파악
           - 문화적 배경이 갈등에 미치는 영향
        
        2. 효과적인 문제 해결 커뮤니케이션
           - 상황별 적절한 대화 접근법
           - 감정 관리 및 명확한 의사 전달 기술
           - 갈등 완화 및 관계 회복 대화법
        
        3. 문제 이후 관계 및 평판 관리
           - 신뢰 회복 및 관계 재구축 전략
           - 조직 내 평판 관리 및 지지 확보
           - 장기적 관계 유지를 위한 후속 조치
        
        4. 조직 문화 내에서의 해결 방안
           - 공식/비공식 문화적 규범 활용
           - 조직 내 지지 네트워크 구축
           - 유사 상황 재발 방지를 위한 문화적 접근
        
        문제 유형: {input_data.get('issue_type', '정보 없음')}
        관계 구조: {input_data.get('relationship', '정보 없음')}
        조직 문화: {input_data.get('workplace_culture', '정보 없음')}
        개인 성향: {input_data.get('personality', '정보 없음')}
        """
    
    def _create_career_workplace_prompt(self, previous_analysis, input_data):
        return f"""
        경력 개발 상황에서의 직장 문화 및 대인관계 관점 조언을 제공해주세요:
        
        1. 성장을 위한 조직 내 네트워킹 전략
           - 핵심 관계망 구축 및 유지 방법
           - 멘토/스폰서 관계 형성 접근법
           - 부서 간 협업을 통한 가시성 확보
        
        2. 조직 문화에 맞는 경력 개발 소통법
           - 성장 의지 표현의 문화적 적절성
           - 직무 변경/승진 논의를 위한 대화 전략
           - 학습 기회 요청의 효과적 방법
        
        3. 다양한 이해관계자와의 관계 관리
           - 상사/부하/동료별 관계 관리 전략
           - 비공식적 영향력 구축 방법
           - 평판 관리 및 브랜딩 접근법
        
        4. 조직 문화 이해를 통한 경력 성장
           - 문화적 성공 요인 파악 및 적용
           - 비공식 규칙 및 관행 활용법
           - 조직 가치와 개인 목표 연계 방법
        
        현재 직급/직무: {input_data.get('position', '정보 없음')}
        경력 목표: {input_data.get('career_goal', '정보 없음')}
        조직 문화: {input_data.get('workplace_culture', '정보 없음')}
        대인관계 스타일: {input_data.get('relationship_style', '정보 없음')}
        """
    
    def _create_general_workplace_prompt(self, previous_analysis, input_data, service_type):
        return f"""
        다음 {service_type} 관련 질문에 대해 직장 문화 및 대인관계 관점에서 종합적인 조언을 제공해주세요:
        
        질문:
        {input_data.get('question', '')}
        
        효과적인 소통 전략, 관계 구축 방법, 문화 적응 기술을 포함한 종합적인 조언을 제공해주세요.
        """


# ============================================================================
# Streamlit 웹 애플리케이션 구현
# ============================================================================

def main():
    """
    메인 함수: Streamlit 웹 애플리케이션의 메인 로직
    """
    # 페이지 기본 설정
    st.set_page_config(
        page_title="AI 인사(HR) 파트너 팀",
        page_icon="👨‍💼👩‍💼📊",
        layout="wide"
    )
    
    # 페이지 제목 및 설명
    st.title("👨‍💼👩‍💼📊 AI 인사(HR) 파트너 팀")
    st.markdown("""
    ### 3명의 전문가가 협업하여 직장인을 위한 맞춤형 인사 관련 조언을 제공합니다
    
    * **김민준 인사 전문가**: 인사 정책, 노동법, 복리후생 등 공식적 제도에 대한 조언
    * **이서연 경력 파트너**: 경력 개발, 역량 향상, 승진/이직 전략에 대한 조언
    * **박지훈 직장 어드바이저**: 직장 문화, 대인관계, 갈등 관리에 대한 조언
    """)
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("🔑 API 설정")
        # API 키 입력 필드 (비밀번호 형식)
        api_key = st.text_input("Google API 키를 입력하세요", type="password")
        
        # API 키가 입력되지 않은 경우 경고 메시지 표시
        if not api_key:
            st.warning("API 키를 입력해주세요.")
            st.stop()
            
        st.markdown("---")
        
        # 전문가 소개
        st.markdown("### 🧠 전문가 소개")
        
        expert_tab = st.selectbox("전문가 정보 보기", 
                                ["김민준 인사 전문가", "이서연 경력 파트너", "박지훈 직장 어드바이저"])
        
        if expert_tab == "김민준 인사 전문가":
            st.markdown("""
            **김민준 인사 전문가**
            
            인사 정책 및 제도 전문가로 15년간 대기업 인사팀 및 노무 컨설팅 경험을 보유하고 있습니다.
            정확한 인사 제도, 노동법, 복리후생 정보를 바탕으로 실용적인 인사 관련 조언을 제공합니다.
            
            * 전문 분야: 인사 정책, 노동법, 급여/복리후생, 평가 시스템, 인사 제도
            * 경력: 대기업 인사팀장, 노무법인 컨설턴트, 인사 시스템 설계 전문가
            * 학력: 인사관리학 석사, 공인노무사 자격증 보유
            """)
        
        elif expert_tab == "이서연 경력 파트너":
            st.markdown("""
            **이서연 경력 파트너**
            
            경력 개발 및 성장 전문가로 12년간 경력 코칭 및 인재 개발 분야에서 활동했습니다.
            개인 맞춤형 경력 경로 설계와 역량 개발 계획을 통해 직장인의 전문적 성장을 지원합니다.
            
            * 전문 분야: 경력 개발, 역량 향상, 승진/이직 전략, 전문성 개발
            * 경력: 커리어 컨설턴트, 기업 HRD 매니저, 리더십 코치
            * 학력: 조직심리학 석사, 공인 커리어 코치 자격증 보유
            """)
        
        elif expert_tab == "박지훈 직장 어드바이저":
            st.markdown("""
            **박지훈 직장 어드바이저**
            
            직장 문화 및 대인관계 전문가로 13년간 조직심리 컨설팅 및 기업 문화 연구 경험을 보유하고 있습니다.
            건강한 직장 생활을 위한 소통 전략, 갈등 관리, 조직 문화 적응 방법에 대한 실질적 조언을 제공합니다.
            
            * 전문 분야: 직장 내 대인관계, 소통 전략, 갈등 관리, 문화 적응
            * 경력: 조직문화 컨설턴트, 기업 교육 강사, 갈등 중재 전문가
            * 학력: 조직심리학 박사, 커뮤니케이션 전문가 자격증 보유
            """)
            
        st.markdown("---")
        # 사용 방법 안내
        st.markdown("### ℹ️ 사용 방법")
        st.markdown("""
        1. API 키를 입력하세요
        2. 원하는 서비스를 선택하세요
        3. 필요한 정보를 입력하세요
        4. '상담 시작' 버튼을 클릭하면 3명의 전문가가 순차적으로 분석합니다
        5. 최종 조언을 확인하세요
        """)
    
    # 서비스 선택 드롭다운
    service = st.selectbox(
        "원하는 서비스를 선택하세요",
        ["고용/계약", "급여/복리후생", "평가/성과", "직장 내 문제", "경력 개발"]
    )
    
    # 카드 스타일 CSS
    st.markdown("""
    <style>
    /* 기본 Streamlit 테마 설정 */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* 기본 텍스트 색상을 흰색으로 설정 */
    .stMarkdown:not(.expert-card), .stText:not(.expert-card) {
        color: #FAFAFA !important;
    }

    /* 답변 카드 스타일 */
    .expert-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        color: #000000;  /* 카드 내부 글자색을 검정색으로 설정 */
        background-color: #FFFFFF;  /* 카드 배경색을 흰색으로 설정 */
    }

    .hr-expert {
        border-left: 5px solid #0077B6;
    }

    .career-expert {
        border-left: 5px solid #2D6A4F;
    }

    .workplace-expert {
        border-left: 5px solid #D4A017;
    }

    /* 답변 카드 내부 텍스트 스타일 */
    .expert-card p, .expert-card li, .expert-card div {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 워크플로우 설명
    with st.expander("에이전틱 워크플로우 프로세스 보기"):
        st.markdown("""
        ### 에이전틱 워크플로우 프로세스
        
        1. **요청 분석**: 사용자의 인사 관련 요청을 분석하여 필요한 전문성 식별
        2. **팀 구성**: 각 요청에 최적화된 AI 인사 전문가 팀 구성
        3. **정책 분석**: 인사 전문가가 관련 정책, 제도, 법규를 분석
        4. **경력 관점**: 경력 파트너가 성장 및 개발 관점에서 조언 추가
        5. **문화/관계 관점**: 직장 어드바이저가 문화 및 대인관계 관점에서 조언 제공
        6. **통합 조언**: 세 전문가의 관점을 통합한 최종 맞춤형 인사 조언 제공
        
        각 전문가는 독립적인 전문성을 가지고 있으며, 순차적 협업을 통해 종합적인 관점을 제공합니다.
        """)
    
    # 선택된 서비스에 따른 UI 표시
    if service == "고용/계약":
        st.subheader("📝 고용/계약 상담")
        
        question = st.text_area("고용/계약 관련 질문을 입력하세요", height=150,
                              placeholder="예: 수습 기간에 어떤 권리가 있나요? 계약서에 없는 업무를 요구받고 있어요. 근로계약 갱신 협상은 어떻게 하는 것이 좋을까요?")
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.text_input("현재 직급/직무", placeholder="예: 대리/마케팅")
            experience = st.selectbox("경력 기간", ["신입", "1-3년", "4-7년", "8-10년", "10년 이상"])
        with col2:
            industry = st.text_input("근무 산업/업종", placeholder="예: IT/금융/제조/서비스")
            company_size = st.selectbox("회사 규모", ["스타트업(30인 미만)", "중소기업(30-300인)", "중견기업(300-1000인)", "대기업(1000인 이상)"])
        
        # 분석 시작 버튼
        if st.button("상담 시작", key="employment_consultation"):
            if question:
                # 코치 팀 초기화
                hr_team = HRPartnerTeam(api_key)
                
                # 입력 데이터 구성
                input_data = {
                    "question": question,
                    "position": position,
                    "experience": experience,
                    "industry": industry,
                    "company_size": company_size
                }
                
                # 결과 처리
                result = hr_team.get_hr_advice("고용/계약", input_data)
                
                # 결과 표시
                st.markdown("### 📊 전문가 분석 결과")
                st.markdown(f"""<div class="expert-card hr-expert"><b>김민준 인사 전문가</b><br><br>{result['hr']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card career-expert"><b>이서연 경력 파트너</b><br><br>{result['career']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card workplace-expert"><b>박지훈 직장 어드바이저 (최종 통합 조언)</b><br><br>{result['workplace']}</div>""", unsafe_allow_html=True)
            else:
                st.warning("질문을 입력해주세요.")
                
    elif service == "급여/복리후생":
        st.subheader("💰 급여/복리후생 상담")
        
        question = st.text_area("급여/복리후생 관련 질문을 입력하세요", height=150,
                              placeholder="예: 적정 연봉 수준은 얼마인가요? 복리후생 협상은 어떻게 하나요? 성과급 기준이 불공정해 보입니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.text_input("현재 직급/직무", placeholder="예: 과장/개발자")
            experience = st.selectbox("경력 기간", ["신입", "1-3년", "4-7년", "8-10년", "10년 이상"])
            salary = st.text_input("현재 급여 수준(선택사항)", placeholder="예: 4500만원/연")
        with col2:
            industry = st.text_input("근무 산업/업종", placeholder="예: IT/금융/제조/서비스")
            location = st.text_input("근무 지역", placeholder="예: 서울/경기/부산")
            company_size = st.selectbox("회사 규모", ["스타트업(30인 미만)", "중소기업(30-300인)", "중견기업(300-1000인)", "대기업(1000인 이상)"])
        
        # 분석 시작 버튼
        if st.button("상담 시작", key="compensation_consultation"):
            if question:
                # 코치 팀 초기화
                hr_team = HRPartnerTeam(api_key)
                
                # 입력 데이터 구성
                input_data = {
                    "question": question,
                    "position": position,
                    "experience": experience,
                    "salary": salary,
                    "industry": industry,
                    "location": location,
                    "company_size": company_size
                }
                
                # 결과 처리
                result = hr_team.get_hr_advice("급여/복리후생", input_data)
                
                # 결과 표시
                st.markdown("### 📊 전문가 분석 결과")
                st.markdown(f"""<div class="expert-card hr-expert"><b>김민준 인사 전문가</b><br><br>{result['hr']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card career-expert"><b>이서연 경력 파트너</b><br><br>{result['career']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card workplace-expert"><b>박지훈 직장 어드바이저 (최종 통합 조언)</b><br><br>{result['workplace']}</div>""", unsafe_allow_html=True)
            else:
                st.warning("질문을 입력해주세요.")
    
    elif service == "평가/성과":
        st.subheader("📊 평가/성과 상담")
        
        question = st.text_area("평가/성과 관련 질문을 입력하세요", height=150,
                              placeholder="예: 불공정한 평가를 받았을 때 어떻게 대응해야 하나요? 성과 목표 설정은 어떻게 하나요? 저성과자로 분류되었습니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.text_input("현재 직급/직무", placeholder="예: 대리/영업")
            performance_concern = st.text_area("평가 관련 고민", height=100, placeholder="예: 객관적 성과가 좋은데 낮은 평가를 받았음")
        with col2:
            company_size = st.selectbox("회사 규모", ["스타트업(30인 미만)", "중소기업(30-300인)", "중견기업(300-1000인)", "대기업(1000인 이상)"])
            evaluation_culture = st.selectbox("평가 문화", ["상대평가 중심", "절대평가 중심", "MBO 방식", "360도 평가", "수시 피드백", "잘 모르겠음"])
        
        career_goal = st.text_input("경력 목표", placeholder="예: 2년 내 팀장 승진, 다른 부서로 이동, 전문성 강화")
        
        # 분석 시작 버튼
        if st.button("상담 시작", key="performance_consultation"):
            if question:
                # 코치 팀 초기화
                hr_team = HRPartnerTeam(api_key)
                
                # 입력 데이터 구성
                input_data = {
                    "question": question,
                    "position": position,
                    "performance_concern": performance_concern,
                    "company_size": company_size,
                    "evaluation_culture": evaluation_culture,
                    "career_goal": career_goal
                }
                
                # 결과 처리
                result = hr_team.get_hr_advice("평가/성과", input_data)
                
                # 결과 표시
                st.markdown("### 📊 전문가 분석 결과")
                st.markdown(f"""<div class="expert-card hr-expert"><b>김민준 인사 전문가</b><br><br>{result['hr']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card career-expert"><b>이서연 경력 파트너</b><br><br>{result['career']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card workplace-expert"><b>박지훈 직장 어드바이저 (최종 통합 조언)</b><br><br>{result['workplace']}</div>""", unsafe_allow_html=True)
            else:
                st.warning("질문을 입력해주세요.")
    
    elif service == "직장 내 문제":
        st.subheader("⚔️ 직장 내 문제 상담")
        
        question = st.text_area("직장 내 문제 관련 질문을 입력하세요", height=150,
                              placeholder="예: 상사와 갈등이 있습니다. 직장 내 괴롭힘을 겪고 있어요. 동료의 업무 태도가 문제입니다.")
        
        issue_type = st.selectbox(
            "문제 유형",
            ["상사와의 갈등", "동료와의 갈등", "부하직원과의 갈등", "직장 내 괴롭힘/불링", "차별/불공정 대우", 
             "업무 분담 문제", "커뮤니케이션 이슈", "팀 내 정치적 문제", "조직 문화 충돌", "워라밸 이슈", "기타"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.text_input("현재 직급/직무", placeholder="예: 주임/연구원")
            experience = st.selectbox("경력 기간", ["신입", "1-3년", "4-7년", "8-10년", "10년 이상"])
            
        with col2:
            company_size = st.selectbox("회사 규모", ["스타트업(30인 미만)", "중소기업(30-300인)", "중견기업(300-1000인)", "대기업(1000인 이상)"])
            workplace_culture = st.selectbox("직장 문화 특성", ["수직적/보수적", "수평적/자유로운", "성과 중심적", "관계 중심적", "혼합형", "잘 모르겠음"])
        
        relationship = st.text_area("관련된 인간관계 구조", height=100, 
                                  placeholder="예: 팀장(50대 남성, 권위적)과의 갈등, 동료 3명 중 2명은 나를 지지하는 편")
        
        personality = st.selectbox("본인의 대인관계 스타일", 
                                ["갈등 회피형", "타협 추구형", "직설적 표현형", "분석적 접근형", "관계 중시형", "과업 중시형"])
    
    # 분석 시작 버튼
    if st.button("상담 시작", key="workplace_issue_consultation"):
        if question:
            # 코치 팀 초기화
            hr_team = HRPartnerTeam(api_key)
            
            # 입력 데이터 구성
            input_data = {
                "question": question,
                "issue_type": issue_type,
                "position": position,
                "experience": experience,
                "company_size": company_size,
                "workplace_culture": workplace_culture,
                "relationship": relationship,
                "personality": personality
            }
            
            # 결과 처리
            result = hr_team.get_hr_advice("직장 내 문제", input_data)
            
            # 결과 표시
            st.markdown("### 📊 전문가 분석 결과")
            st.markdown(f"""<div class="expert-card hr-expert"><b>김민준 인사 전문가</b><br><br>{result['hr']}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="expert-card career-expert"><b>이서연 경력 파트너</b><br><br>{result['career']}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="expert-card workplace-expert"><b>박지훈 직장 어드바이저 (최종 통합 조언)</b><br><br>{result['workplace']}</div>""", unsafe_allow_html=True)
        else:
            st.warning("질문을 입력해주세요.")

    # 경력 개발 서비스 UI 부분
    elif service == "경력 개발":
        st.subheader("🚀 경력 개발 상담")
        
        question = st.text_area("경력 개발 관련 질문을 입력하세요", height=150,
                              placeholder="예: 현 직장에서 성장 가능성이 있을까요? 이직을 고민 중입니다. 승진을 위해 어떤 역량을 개발해야 할까요?")
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.text_input("현재 직급/직무", placeholder="예: 과장/기획")
            experience = st.selectbox("경력 기간", ["신입", "1-3년", "4-7년", "8-10년", "10년 이상"])
            
        with col2:
            career_goal = st.text_area("경력 목표", height=100, placeholder="예: 2년 내 팀장 승진, 1년 내 데이터 분석 직무로 이직, 전문성 강화")
            relationship_style = st.selectbox("대인관계 스타일", ["네트워킹 중시형", "소수 관계 집중형", "업무 중심 관계형", "멘토링 선호형"])
        
        skill_interests = st.text_area("관심 있는 역량/스킬 분야", height=100, 
                                    placeholder="예: 데이터 분석, 프로젝트 관리, 리더십, 프레젠테이션 스킬")
        
        workplace_culture = st.selectbox("현 직장 문화 특성", ["수직적/보수적", "수평적/자유로운", "성과 중심적", "관계 중심적", "혼합형", "잘 모르겠음"])
        
        # 분석 시작 버튼
        if st.button("상담 시작", key="career_consultation"):
            if question:
                # 코치 팀 초기화
                hr_team = HRPartnerTeam(api_key)
                
                # 입력 데이터 구성
                input_data = {
                    "question": question,
                    "position": position,
                    "experience": experience,
                    "career_goal": career_goal,
                    "relationship_style": relationship_style,
                    "skill_interests": skill_interests,
                    "workplace_culture": workplace_culture
                }
                
                # 결과 처리
                result = hr_team.get_hr_advice("경력 개발", input_data)
                
                # 결과 표시
                st.markdown("### 📊 전문가 분석 결과")
                st.markdown(f"""<div class="expert-card hr-expert"><b>김민준 인사 전문가</b><br><br>{result['hr']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card career-expert"><b>이서연 경력 파트너</b><br><br>{result['career']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="expert-card workplace-expert"><b>박지훈 직장 어드바이저 (최종 통합 조언)</b><br><br>{result['workplace']}</div>""", unsafe_allow_html=True)
            else:
                st.warning("질문을 입력해주세요.")
    
    # 추가 도구 섹션 (선택적으로 표시)
    st.markdown("## 💼 추가 셀프 서비스 도구")
    add_extension_tabs()


# 급여 분석 도구 (간소화 버전)
def analyze_salary(industry, position, experience, location):
    """
    급여 데이터 분석 및 시각화 함수 (가상 데이터 사용)
    """
    # 가상 데이터 생성
    np.random.seed(42)
    
    data = {
        "산업": np.random.choice(["IT", "금융", "제조", "서비스", "의료", "교육"], 100),
        "직급": np.random.choice(["사원/주임", "대리", "과장", "차장", "부장"], 100),
        "경력": np.random.choice(["1-3년", "4-7년", "8-10년", "10년 이상"], 100),
        "지역": np.random.choice(["서울", "경기", "부산", "기타"], 100),
        "연봉(만원)": np.random.normal(4500, 1500, 100).astype(int)
    }
    
    df = pd.DataFrame(data)
    
    # 필터링
    filtered_df = df
    if industry:
        filtered_df = filtered_df[filtered_df["산업"] == industry]
    if position:
        filtered_df = filtered_df[filtered_df["직급"] == position]
    if experience:
        filtered_df = filtered_df[filtered_df["경력"] == experience]
    if location:
        filtered_df = filtered_df[filtered_df["지역"] == location]
    
    # 시각화 및 결과 계산
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(filtered_df) > 0:
        sns.boxplot(data=filtered_df, x="직급", y="연봉(만원)", hue="산업", ax=ax)
        ax.set_title("직급/산업별 급여 분포")
        avg_salary = filtered_df["연봉(만원)"].mean()
        min_salary = filtered_df["연봉(만원)"].min()
        max_salary = filtered_df["연봉(만원)"].max()
        
        result = {
            "avg_salary": avg_salary,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "fig": fig
        }
    else:
        result = {
            "avg_salary": 0,
            "min_salary": 0,
            "max_salary": 0,
            "fig": None
        }
    
    return result


# 추가 도구 탭 함수 (간소화 버전)
def add_extension_tabs():
    """
    확장 기능 탭 (급여 분석, 직장 문제 해결 가이드, 경력 개발 도구)
    """
    tools_tab1, tools_tab2, tools_tab3 = st.tabs(["💰 급여 분석 도구", "🛠️ 직장 문제 해결 가이드", "🚀 경력 개발 도구"])
    
    with tools_tab1:
        # 급여 분석 도구 UI
        st.markdown("## 📊 급여 통계 및 시장 분석")
        st.write("본인의 급여 수준이 시장 대비 적정한지 확인하고 협상에 필요한 데이터를 확인해보세요.")
        
        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox(
                "산업/업종",
                ["", "IT/소프트웨어", "금융/보험", "제조/생산", "서비스업", "의료/제약", "교육"]
            )
            position = st.selectbox(
                "직급/직책",
                ["", "사원/주임급", "대리급", "과장급", "차장급", "부장급"]
            )
        with col2:
            experience = st.selectbox(
                "경력 기간",
                ["", "신입", "1-3년", "4-7년", "8-10년", "10년 이상"]
            )
            location = st.selectbox(
                "근무 지역",
                ["", "서울", "경기", "인천", "부산/경남", "대구/경북", "광주/전라", "대전/충청"]
            )
        
        if st.button("급여 통계 분석"):
            with st.spinner("급여 데이터를 분석 중입니다..."):
                # 실제 분석 함수 호출
                result = analyze_salary(industry, position, experience, location)
                
                # 결과 표시 (간소화)
                if result["avg_salary"] > 0:
                    st.success(f"분석이 완료되었습니다! 해당 조건의 평균 연봉은 {int(result['avg_salary']):,}만원입니다.")
                    
                    # 통계 표시
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("평균 연봉", f"{int(result['avg_salary']):,}만원")
                    with stat_col2:
                        st.metric("최소 연봉", f"{int(result['min_salary']):,}만원")
                    with stat_col3:
                        st.metric("최대 연봉", f"{int(result['max_salary']):,}만원")
                    
                    # 그래프 표시
                    if result["fig"]:
                        st.pyplot(result["fig"])
                else:
                    st.warning("해당 조건에 맞는 데이터가 충분하지 않습니다. 조건을 변경해보세요.")
    
    with tools_tab2:
        # 직장 내 문제 해결 가이드 (간소화)
        st.markdown("## 🛠️ 직장 내 문제 해결 가이드")
        st.write("직장에서 발생할 수 있는 다양한 문제 상황에 대한 자가진단 및 해결 가이드")
        
        issue_category = st.selectbox(
            "문제 유형 선택",
            ["직장 내 갈등", "직장 내 괴롭힘/불링", "불공정한 업무 분배", "상사와의 소통 문제", "팀 내 협업 이슈"]
        )
        
        st.markdown("#### 문제 상황 자가진단")
        # 간소화된 자가진단 (실제 구현 시 확장 필요)
        questions = [
            "해당 문제로 인해 업무 수행에 지장이 있나요?",
            "문제가 한 달 이상 지속되고 있나요?",
            "문제 해결을 위해 노력해본 적이 있나요?",
            "이 문제로 인해 건강이나 정신적 웰빙에 영향을 받고 있나요?",
            "문제 상황이 점점 악화되고 있나요?"
        ]
        
        st.markdown("#### 해결 가이드")
        st.markdown("""
        **단계적 문제 해결 접근법**:
        
        1. **상황 정의 및 분석**: 문제 상황을 객관적으로 기록하고 분석
        2. **해결 방향 설정**: 원하는 결과와 수용 가능한 대안 정의
        3. **소통 전략 수립**: 적절한 시점, 장소, 대화 방식 계획
        4. **대화 및 협상**: 감정이 아닌 사실 기반 대화와 해결책 모색
        5. **조직 자원 활용**: 필요시 인사팀, 멘토, 상급자 지원 요청
        6. **후속 조치 및 관계 관리**: 이후 관계 관리 및 유사 상황 예방
        """)
    
    with tools_tab3:
        # 경력 개발 도구 (간소화)
        st.markdown("## 🚀 경력 개발 플래너")
        st.write("현재 역량을 진단하고 목표 달성을 위한 맞춤형 경력 개발 계획을 수립해보세요.")
        
        tabs = st.tabs(["역량 진단", "목표 설정", "개발 계획"])
        
        with tabs[0]:
            st.markdown("#### 역량 자가진단")
            st.write("각 역량별로 현재 수준을 평가해주세요 (1: 매우 낮음, 5: 매우 높음)")
            
            # 샘플 역량 리스트 (실제 구현 시 직무별 맞춤화 필요)
            skills = ["전문 지식", "의사소통", "리더십", "문제 해결", "팀워크", "적응력"]
            
            col1, col2 = st.columns(2)
            skill_ratings = {}
            
            for i, skill in enumerate(skills):
                if i < len(skills) // 2:
                    with col1:
                        skill_ratings[skill] = st.slider(skill, 1, 5, 3, key=f"skill_{skill}")
                else:
                    with col2:
                        skill_ratings[skill] = st.slider(skill, 1, 5, 3, key=f"skill_{skill}")
        
        with tabs[1]:
            st.markdown("#### 경력 목표 설정")
            
            goal_timeframe = st.radio("목표 기간", ["단기 (1년 이내)", "중기 (1-3년)", "장기 (3-5년)"])
            
            goal_description = st.text_area("구체적인 목표 설명", 
                                            placeholder="예: 2년 내 프로젝트 매니저로 승진하고 5명 이상의 팀을 리드하고 싶습니다.")
        
        with tabs[2]:
            st.markdown("#### 개발 계획 수립")
            
            st.markdown("""
            **효과적인 경력 개발 계획 요소**:
            
            1. **구체적 목표 분해**: 큰 목표를 작은 마일스톤으로 분해
            2. **핵심 역량 식별**: 목표 달성에 필요한 역량 명확화
            3. **학습 자원 확보**: 적절한 교육, 멘토링, 경험적 학습 기회 활용
            4. **실행 계획 수립**: 시간, 우선순위, 리소스 계획
            5. **진척도 측정**: 정기적 진행 상황 검토 및 조정
            """)
            
            st.info("개발 계획을 생성하려면 역량 진단과 목표 설정을 완료한 후 '개발 계획 생성' 버튼을 클릭하세요.")


# 스크립트가 직접 실행될 때만 main() 함수 실행
if __name__ == "__main__":
    main()
