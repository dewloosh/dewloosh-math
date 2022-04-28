# -*- coding: utf-8 -*-
import codecs
import os.path
from setuptools import find_namespace_packages, setup
from setuptools.command.install import install


_AxisVM_version_ = None

class InstallCommand(install):
    user_options = install.user_options + [
        ('version=', None, 'AxisVM version')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.version = None

    def finalize_options(self):
        #print('The custom option for install is ', self.custom_option)
        install.finalize_options(self)

    def run(self):
        global _AxisVM_version_
        _AxisVM_version_ = self.version
        #install.run(self)  # OR: install.do_egg_install(self)
		if not self._called_from_setup(inspect.currentframe()):
			orig.install.run(self)
		else:
			self.do_egg_install()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
    
    
def get_description(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__description__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find description string.")


with open("README.md", "r") as fh:
    long_description = fh.read()


with open('requirements.txt') as f:
    required = f.read().splitlines()


_module = os.listdir(os.path.join(os.path.dirname(__file__), "src\\dewloosh"))[0]
_init_path = "src/dewloosh/{}/__init__.py".format(_module)
_version = get_version(_init_path)
_description = get_description(_init_path)
_url = 'https://github.com/dewloosh/dewloosh-{}'.format(_module)
_download_url = _url + '/archive/refs/tags/{}.zip'.format(_version)


setup(
	name="dewloosh.{}".format(_module),
    version=_version,                        
    author="dewloosh",
    author_email = 'dewloosh@gmail.com',                   
    description=_description,
    long_description=long_description,   
    long_description_content_type="text/markdown",
	url = _url, 
    download_url = _download_url,
	packages=find_namespace_packages(where='src', include=['dewloosh.*']),
	classifiers=[
        'Development Status :: 3 - Alpha',     
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3',
		'Operating System :: OS Independent'
    ],                                      
    python_requires='>=3.6',                             
    package_dir={'':'src'},     
    install_requires=required,
	zip_safe=False,
	cmdclass={
        'install': InstallCommand,
    },
)

