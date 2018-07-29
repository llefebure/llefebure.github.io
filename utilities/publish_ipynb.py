# Script for converting ipynb into markdown formatted to display on the jekyll site
# Should be run from the same directory as the ipynb that is being converted.
# Note that if the post doesn't have additional figures, etc, the last two
# bash commands will fail, but that's okay.
import argparse
import datetime
import os

STEM = "/Users/lukelefebure/Documents/Projects/llefebure.github.io"
POST_DIR = "/_posts/"
PLOT_DIR = "/assets/Pyfig/"

def convert(args):
	today = datetime.date.today()
	post_name = args.ipynb.split("/")[-1].replace(".ipynb", "")
	tmp_dir = post_name + "_files" # dir created by nbconvert
	post_nm = STEM + POST_DIR + str(today) + "-" + post_name + ".md" # final post name
	cmds = [
		"jupyter nbconvert --to markdown {f}".format(f = args.ipynb),
		"mv {md} {post_nm}".format(md = post_name + ".md", post_nm = post_nm),
		"cp ./{tmp_dir}/* {plot_dir}".format(tmp_dir = tmp_dir, plot_dir = STEM + PLOT_DIR),
		"rm -rf ./{tmp_dir}/".format(tmp_dir = tmp_dir)
	]
	for cmd in cmds:
		os.system(cmd)
	return post_nm, tmp_dir

def makeHeader(args):
	header = '''---
layout: post
title: "{title}"
excerpt: "{excerpt}"
categories: [{categories}]
comments: true
---'''
	
	return header.format(title = args.title, excerpt = args.excerpt, categories = ", ".join(args.categories))

def rewriteFile(fn, tmp_dir, header):
	with open(fn, "r") as f:
		content = f.read()
		content = content.replace(tmp_dir, PLOT_DIR[:-1])
	with open(fn, "w") as f:
		f.write(header + "\n" + content)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ipynb", help = "Path to ipynb", type = str)
	parser.add_argument("--title", help = "Title of post", type = str)
	parser.add_argument("--excerpt", help = "Excerpt summary to show on homepage, archive, etc", type = str)
	parser.add_argument("--categories", help = "Excerpt summary to show on homepage, archive, etc", nargs = "+")
	args = parser.parse_args()
	md_file, tmp_dir = convert(args)
	header = makeHeader(args)
	rewriteFile(md_file, tmp_dir, header)