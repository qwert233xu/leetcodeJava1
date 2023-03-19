package com.wenqu.leetcode;


import org.junit.Test;

import java.util.*;

/**
 * 回溯算法
 * Back Track Test
 */

public class backTrackTest {

    /*
        输入：candidates = [2,3,6,7], target = 7
        输出：[[2,2,3],[7]]   candidates不可重复，元素可重复选取
     */
    @Test
    public void test01(){
        int[] intPut = {2, 3, 6, 7};
        int targets = 7;
        List<List<Integer>> res = new ArrayList<>();

        // 加上剪枝
        Arrays.sort(intPut);
        res = combinationSum(intPut, targets);
        System.out.println(res);
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0){
            return res;
        }

        if (candidates.length == 1){
            if (candidates[0] != target){
                return res;
            }else {
                res.add(new ArrayList<>(candidates[0]));
                return res;
            }
        }

        backTrack(res, candidates, target, new ArrayList<Integer>(), 0, candidates.length);
        return res;

    }
    public static void backTrack(List<List<Integer>> res, int[] candidates, int target, ArrayList<Integer> temp, int index, int length){
        if (target == 0){
            res.add(temp);
            return;
        }

        /*
            *** 回溯思想重点理解双循环概念，即小循环和大循环的概念
            1、小循环即寻找单个数字可以满足的前提下（没有超越），其它数字的循环
            2、大循环是在小循环都不满足下，进一步扔掉小循环里最后一个数字，再依次加入其它数字后，继续小循环
            3、实际是按照小循环---大循环交替执行
            排序后则可以在小循环和大循环基础上剪枝
         */
        for (int i = index; i < candidates.length; i++) {
            // 剪枝  建立在有序的基础上
            if (target - candidates[i] < 0){
                break;
            }
            temp.add(candidates[i]);
            backTrack(res, candidates, target - candidates[i], new ArrayList<>(temp), i, candidates.length);
            temp.remove(temp.size() - 1);
            index += 1;
        }

    }

    /*
        输入: candidates = [10,1,2,7,6,1,5], target = 8,
        输出:             candidates可重复，元素不可重复选取
        [
        [1,1,6],
        [1,2,5],
        [1,7],
        [2,6]
        ]
     */
    @Test
    public void test02(){
        int[] intPut = {10,1,2,7,6,1,5};
        int target = 8;
        List<List<Integer>> res = new ArrayList<>();

        // 加上剪枝
        Arrays.sort(intPut);
        res = combinationSum2(intPut, target);
        System.out.println(res);
    }
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {

        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0){
            return res;
        }

        if (candidates.length == 1){
            if (candidates[0] != target){
                return res;
            }else {
                res.add(new ArrayList<>(candidates[0]));
                return res;
            }
        }

        backTrack2(res, candidates, target, new ArrayList<Integer>(), 0, candidates.length);
        return res;
    }

    public static void backTrack2(List<List<Integer>> res, int[] candidates, int target, ArrayList<Integer> temp, int index, int length){
        if (target == 0){
            res.add(temp);
            return;
        }

        for (int i = index; i < length; i++) {
            // 剪枝  建立在有序的基础上
            if (target - candidates[i] < 0){
                break;
            }
            // 小剪枝：同一层相同数值的结点，从第 2 个开始，候选数更少，结果一定发生重复，因此跳过，用 continue    防止重复元素发生重复  （小循环去掉重复）
            if (i > index && candidates[i] == candidates[i - 1]) {
                continue;
            }

            temp.add(candidates[i]);
            backTrack2(res, candidates, target - candidates[i], new ArrayList<>(temp), i + 1, length);    // i+1 同一个元素只能用一次
            temp.remove(temp.size() - 1);

        }

    }

    /*  全排列 1
        给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
        输入：nums = [1,2,3]
        输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
     */
    @Test
    public void test03(){
        int[] intPut = {1, 2, 3};
        List<List<Integer>> res = new ArrayList<>();

        // 加上剪枝
        res = permute(intPut);
        System.out.println(res);

    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();

        if (nums.length == 0){
            return res;
        }

        boolean[] used = new boolean[nums.length];
        backTrack3(res, nums, new ArrayList<Integer>(), 0, nums.length, used);

        return res;

    }

    public static void backTrack3(List<List<Integer>> res, int[] nums, ArrayList<Integer> temp, int begin, int length, boolean[] used){

        if (begin == length){
            res.add(new ArrayList<>(temp));
            return;
        }

        for (int i = 0; i < length; i++) {   // i 从 0 开始的目的是顺序不一样的组合可认为是不同排列
            if (!used[i]) {  // 保证小循环里不重复使用自己
                temp.add(nums[i]);
                used[i] = true;   // 小循环里不会用   使用过则为  true
                backTrack3(res, nums, temp, begin + 1, length, used);  // begin + 1 保证不重复使用自己
                temp.remove(temp.size() - 1);
                used[i] = false;  // 大循环里恢复 为 false
            }
        }
    }

    /*
        全排列 2
        给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
        输入：nums = [1,1,2]
        输出：
        [[1,1,2],
         [1,2,1],
         [2,1,1]]
     */
    @Test
    public void test04(){

        int[] intPut = {1, 1, 2};
        List<List<Integer>> res = new ArrayList<>();

        Arrays.sort(intPut);
        res = permuteUnique(intPut);
        System.out.println(res);


    }
    public List<List<Integer>> permuteUnique(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();

        if (nums.length == 0){
            return res;
        }

        boolean[] used = new boolean[nums.length];
        backTrack4(res, nums, new ArrayList<Integer>(), 0, nums.length, used);

        return res;


    }

    public static void backTrack4(List<List<Integer>> res, int[] nums, ArrayList<Integer> temp, int begin, int length, boolean[] used){

        if (begin == length){
            res.add(new ArrayList<>(temp));
            return;
        }

        for (int i = 0; i < length; i++) {   // i 从 0 开始的目的是顺序不一样的组合可认为是不同排列

            if (!used[i]) {  // 保证小循环里不重复使用自己
                // 剪枝条件：i > 0 是为了保证 nums[i - 1] 有意义
                // 写 !used[i - 1] 是因为 nums[i - 1] 在深度优先遍历的过程中刚刚被撤销选择
                if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                    continue;
                }
                temp.add(nums[i]);
                used[i] = true;   // 小循环里不会用   使用过则为  true
                backTrack4(res, nums, temp, begin + 1, length, used);  // begin + 1 保证不重复使用自己
                temp.remove(temp.size() - 1);
                used[i] = false;  // 大循环里恢复 为 false
            }
        }
    }

    /*
        给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
        给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
        输入：digits = "23"
        输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
     */
    @Test
    public void test05(){

        String intPut = "8";

        List<String> res = new ArrayList<>();

        res = letterCombinations(intPut);
        System.out.println(res);

    }
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();

        if (digits.length() == 0){
            return res;
        }

        // 映射表
        HashMap<String, String> map = new HashMap<>();

        map.put("2", new String("abc"));
        map.put("3", new String("def"));
        map.put("4", new String("ghi"));
        map.put("5", new String("jkl"));
        map.put("6", new String("mno"));
        map.put("7", new String("pqrs"));
        map.put("8", new String("tuv"));
        map.put("9", new String("xwyz"));

        backTrack5(res, 0, "", digits, map);

        return res;

    }

    public static void backTrack5(List<String> res, int index, String s, String digits, HashMap<String, String> map){

        if (s.length() == digits.length()){
            res.add(s);
            return;
        }

        Character c = digits.charAt(index);
        String letters = map.get(c + "");
        for (int i = 0; i < letters.length(); i++) {
            backTrack5(res, index + 1, s + letters.charAt(i) + "", digits, map);
        }

    }
}



