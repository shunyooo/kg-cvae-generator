import re
import pandas
import tqdm
import codecs
import json

from konlpy.tag import Mecab

default_dialog_path = '/scatter/pingpong/corpus/new_pingpong_corpus/pingpong_corpus.{0:04d}.txt'


class Pingpong2JsonConverter:
    def __init__(self):
        self.kakao_file_pattern_str = \
            '.*[0-9]{4,}_([A-Z]+?_[A-Z]+?_[A-Z]+?_[A-Z]+?_[A-Z]+?)_[A-Z]+?\.log'
        self.kakao_file_pattern = re.compile(self.kakao_file_pattern_str)

        self.meta_sex_dict = {'FEMALE': 0, 'MALE': 1}
        self.meta_age_dict = {'STUDENT': 0, 'COLLEGIAN': 1, 'CIVILIAN': 2}
        self.meta_relation_dict = {'FRIEND': 0, 'LOVER': 1}

        self.tokenizer = Mecab()

    def _read_meta_user_key(self, path_idx_dict_file_path, end_line, is_save=False):
        """
        원 파일 목록 파일을 읽어서 각 파일에 대한 메타 정보를 사전으로 저장하는 함수.

        path_idx_dict_file: (원 파일 번호) \t (원 파일 이름)

        :param path_idx_dict_file_path: 파일 경로.
        :param end_line: path_idx_dict의 파일 읽기 줄수 한도.
        :param is_save: pandas dict 형태로 메타 정보를 저장할지 유무.
        :return:
        """
        path_idx_dict_file_reader = open(path_idx_dict_file_path, 'r')
        meta_user_key_dict = {}
        pandas_list = []
        for line_idx, line in enumerate(tqdm.tqdm(path_idx_dict_file_reader)):
            if line_idx >= end_line:
                break
            else:
                dialog_idx_str, orig_file_path_string = line.strip().split('\t')
                dialog_idx = int(dialog_idx_str)
                is_matched, meta_info = self._read_meta_from_path(orig_file_path_string)
                if is_matched:
                    meta_user_key_dict[dialog_idx] = meta_info
                    pandas_dict = {"dialog_idx": dialog_idx}
                    pandas_dict.update(meta_info)
                    pandas_list.append(pandas_dict)

        df = pandas.DataFrame(pandas_list)
        df = df.set_index('dialog_idx')
        if is_save:
            df.to_csv('meta_user_key.csv', index=False)
        return meta_user_key_dict, df

    def _read_meta_from_path(self, file_path_string):
        """
        파일 이름을 보고 메타 정보를 추출하는 함수.
        실패시, False를 리턴함.

        :param file_path_string: 파일 이름.
        :return: (매칭 여부, 메타 데이터 사전)
        """
        matched = False
        meta_dict = {}

        if self.kakao_file_pattern.search(file_path_string):
            matched = True
            meta = self.kakao_file_pattern.sub(r'\1', file_path_string)
            my_sex, my_age, your_sex, your_age, relation = meta.split("_")
            meta_dict["my_sex"] = my_sex
            meta_dict["int_my_sex"] = self.meta_sex_dict.get(my_sex, 0)
            meta_dict["my_age"] = my_age
            meta_dict["int_my_age"] = self.meta_age_dict.get(my_age, 2)

            meta_dict["your_sex"] = your_sex
            meta_dict["int_your_sex"] = self.meta_sex_dict.get(your_sex, 0)
            meta_dict["your_age"] = your_age
            meta_dict["int_your_age"] = self.meta_age_dict.get(your_age, 2)

            meta_dict["relation"] = relation
            meta_dict["int_relation"] = self.meta_relation_dict.get(relation, 0)

        return matched, meta_dict

    @staticmethod
    def _read_matching(matching_path, start_line, end_line,
                       min_session_length, meta_user_key, session_limit,
                       session_print_num, margin=100):
        """
        matching.txt: (원 파일 번호) \t (세션 번호 0x000 00000) \t (세션 길이)

        :param matching_path: matching의 경로.
        :param start_line: matching의 파일 읽기 시작줄 번호.
        :param end_line: matching의 파일 읽기 끝줄 번호.
        :param min_session_length: 최소 세션 길이.
        :param meta_user_key: {원 파일 번호: (메타 정보)}
        :param session_limit: 세션 갯수 제한.
        :param session_print_num: 세션 인쇄.
        :param margin: 파일 최대 번호 마진.
        :return:
        """
        matching_reader = open(matching_path, 'r')
        max_dialog_num = max(meta_user_key)
        result_dict = {}
        case_num = 0
        for line_idx, line in enumerate(matching_reader):
            if line_idx < start_line:
                continue
            if line_idx >= end_line:
                break
            # 데이터 형식: dialog_idx \t session_idx \t session_length
            sline = line.strip()
            dialog_idx_str, session_idx_str, session_length_str = sline.split('\t')
            dialog_idx = int(dialog_idx_str)
            if dialog_idx > max_dialog_num + margin:
                break

            session_length = int(session_length_str)

            # session_length 가 min 보다 큰 경우만 채택
            if int(session_length) > min_session_length:
                if int(dialog_idx) not in meta_user_key:
                    continue
                else:
                    case_num += 1
                    file_number = int(session_idx_str[:5], 16)
                    meta_info = meta_user_key[dialog_idx]
                    session_info = {"dialog_idx": dialog_idx,
                                    "session_idx_str": session_idx_str,
                                    "session_length": session_length}
                    if file_number not in result_dict:
                        result_dict[file_number] = []

                    result_dict[file_number].append((session_info, meta_info))

                    if (case_num % session_print_num) == 0:
                        print("Session Extracted: {0} from line {1}".format(case_num,
                                                                            line_idx))

            if case_num >= session_limit:
                break

        return result_dict

    def _read_session_with_dialog_idx(self, file_number, session_meta_info_pairs,
                                      file_format_path=default_dialog_path):
        """
        (세션 번호) \t RECV \t TIME \t CONCAT \t MESG
        0x0000001e	RECV	20140528010100	1	언제까지 하는데?

        :param file_number:
        :param session_meta_info_pairs:
        :param file_format_path:
        :return:
        """
        session_idx_pair_dict = {}
        session_idx_strs = []
        for session_meta_info_pair in session_meta_info_pairs:
            session_info, meta_info = session_meta_info_pair
            session_idx_str = session_info['session_idx_str']
            session_idx_pair_dict[session_idx_str] = session_meta_info_pair
            session_idx_strs.append(session_idx_str)
        session_idx_strs.sort()
        file_path = file_format_path.format(file_number)
        file_reader = codecs.open(file_path, 'r', encoding='utf-8')
        current_marker = 0
        session_read_started = False
        marker_limit = len(session_idx_strs)
        overall_utts = {}
        utts = []

        for line_idx, line in enumerate(file_reader):
            if current_marker >= marker_limit:
                # print(current_marker, marker_limit)
                break
            else:
                line_fields = line.strip().split("\t")
                line_session_idx_str = line_fields[0]
                mesg = line_fields[4]
                current_turn = "A" if line_fields[1] == "SEND" else "B"
                if session_idx_strs[current_marker] == line_session_idx_str:
                    dialog_turn_tuple = (current_turn,
                                         ["<s>"] + self.tokenizer.morphs(mesg) + ["</s>"], mesg)
                    if not session_read_started:
                        session_read_started = True
                        utts = [dialog_turn_tuple]
                    else:
                        utts.append(dialog_turn_tuple)
                else:
                    if session_read_started:
                        session_read_started = False
                        if len(utts) != 0:
                            overall_utts[session_idx_strs[current_marker]] = utts
                        current_marker += 1
                        utts = []
                    else:
                        pass
        if session_read_started and current_marker < marker_limit:
            session_read_started = False
            if len(utts) != 0:
                overall_utts[session_idx_strs[current_marker]] = utts
            utts = []
        # print(session_idx_strs)
        # print(list(overall_utts.keys()))

        overall_result = []
        for session_idx_str in session_idx_strs:
            session_info, meta_info = session_idx_pair_dict[session_idx_str]
            session_dict = dict()
            session_dict['session_idx_str'] = session_idx_str
            session_dict['A'] = {'age': meta_info['int_my_age'], 'age_group': meta_info["my_age"],
                                 'sex': meta_info['int_my_sex'], 'sex_group': meta_info["my_sex"],
                                 'relation': meta_info['int_relation'],
                                 'relation_group': meta_info['relation']}

            session_dict['B'] = {'age': meta_info['int_your_age'],
                                 'age_group': meta_info["your_age"],
                                 'sex': meta_info['int_your_sex'],
                                 'sex_group': meta_info["your_sex"],
                                 'relation': meta_info['int_relation'],
                                 'relation_group': meta_info['relation']}

            session_dict['topic'] = "핑퐁대화"
            session_dict['prompt'] = "자유대화를 나눠보세요."
            session_dict['utts'] = overall_utts[session_idx_str]

            overall_result.append(session_dict)

        return overall_result

    def file2json(self, path_idx_dict_file_path, end_line, is_save_meta,
                  matching_path, matching_start_line, matching_end_line,
                  min_session_length, session_limit, session_print_num,
                  output_file_path, file_format_path=default_dialog_path):

        meta_user_key, df = self._read_meta_user_key(path_idx_dict_file_path,
                                                     end_line, is_save_meta)
        result_dict = self._read_matching(matching_path, matching_start_line,
                                          matching_end_line, min_session_length,
                                          meta_user_key, session_limit, session_print_num)

        overall_result = []
        for file_number in result_dict:
            print("File number {0}".format(file_number))
            overall_result.extend(self._read_session_with_dialog_idx(file_number,
                                                                     result_dict[file_number],
                                                                     file_format_path))

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(overall_result, f, ensure_ascii=False, indent=4)
        return overall_result


if __name__=="__main__":
    converter = Pingpong2JsonConverter()
    axp = converter.file2json("/scatter/pingpong/corpus/new_pingpong_corpus/path_idx_dict.txt",
                              100000, False,
                              "/scatter/pingpong/corpus/new_pingpong_corpus/matching.txt", 0,
                              42380374, 11, 100000, 5000,
                              "output.json", file_format_path=default_dialog_path)