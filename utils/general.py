import configparser


class cfg:
    def __init__(self, file: str, encoding: str = 'utf-8'):
        """
        :param file: 文件路径
        :param encoding: 编码，默认utf-8
        """
        self.file = file
        self.encoding = encoding
        self.__conf = configparser.ConfigParser()
        self.__conf.read(self.file, encoding=self.encoding)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def set(self, section, key, value, save: bool = False):
        """
        设置值
        :param section: 小节名
        :param key:键名
        :param value:键值
        :param save: 自动保存，默认关闭
        :return:None
        """
        try:
            self.__conf.add_section(str(section))
        except:
            pass
        self.__conf.set(section=str(section), option=str(key), value=str(value))
        if save:
            self.save()

    def search(self, section, key, default_value: any = None, return_type=str):
        """
        :param return_type:返回类型
        :param section:小节名
        :param key:键名
        :param default_value:当找不到时返回的内容
        :return:返回default，默认None
        """
        try:
            result = self.__conf.get(section=str(section), option=str(key))
            if result == "":
                return default_value
        except:
            return default_value
        if return_type == bool:
            return result.lower() in ('yes', 'true', 't', 'y', '1')
        return return_type(result)

    def items(self, section):
        try:
            res = self.__conf.items(str(section), )
            return res
        except:
            return []

    def section(self) -> list[str]:
        """
        :return: 以列表返回所有键名
        """
        try:
            return self.__conf.sections()
        except:
            return []

    def delete(self, section, *key, save: bool = False):
        """
        删除小节或键值
        :param section:小节名
        :param key:键名，为空则删除小节
        :param save: 自动保存，默认关闭
        :return:None
        """
        try:
            if len(key) > 0:
                for i in key:
                    self.__conf.remove_option(section=section, option=i)
            else:
                self.__conf.remove_section(section=section)
        except:
            pass
        if save:
            self.save()

    def copy(self, old_section, new_section, save: bool = False):
        """
        复制已有小节
        :param old_section: 要复制的小节
        :param new_section: 小节新名字
        :param save: 自动保存，默认关闭
        :return: None
        """
        try:
            dict1 = dict(self.__conf.items(section=old_section))
            for i in dict1:
                self.set(new_section, i, dict1[i])
        except:
            raise ValueError(f'"{old_section}" not found')
        if save:
            self.save()

    def save(self, encoding=None):
        """
        :param encoding: 编码
        :return: None
        """
        with open(self.file, 'w', encoding=encoding if encoding else self.encoding) as conf:
            self.__conf.write(conf)
        del self