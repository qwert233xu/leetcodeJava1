package com.xu;

import java.util.*;

/**
 * 牛客网  Top101
 *
 */
public class App 
{

    public static void main(String[] args )
    {
        System.out.println( "Hello World!" );
    }

    /** 单链表
     * 1.反转链表 递归实现
     * @param root
     * @return
     */
    public ListNode ReverseList(ListNode root) {
        //终止条件
        if (root == null || root.next == null)
            return root;
        //保存当前节点的下一个结点
        ListNode next = root.next;
        //从当前节点的下一个结点开始递归调用
        ListNode reverse = ReverseList(next);
        //reverse是反转之后的链表，因为函数reverseList
        // 表示的是对链表的反转，所以反转完之后next肯定
        // 是链表reverse的尾结点，然后我们再把当前节点
        //head挂到next节点的后面就完成了链表的反转。
        next.next = root;
        //这里head相当于变成了尾结点，尾结点都是为空的，
        //否则会构成环
        root.next = null;
        return reverse;
    }
    /**
     * 1.反转链表 栈实现
     * @param root
     * @return
     */
    public ListNode ReverseList2(ListNode root) {
        if (root == null || root.next == null) return root;

        Stack<ListNode> stack = new Stack<>();

        while (root.next != null){
            stack.push(root);
            root = root.next;
        }

        ListNode temp = root;
        while (!stack.isEmpty()){
            temp.next = stack.peek();
            stack.pop();
            temp.next.next = null;
            temp = temp.next;
        }
        return root;
    }

    /**
     * 1.反转链表 双链表实现
     * @param root
     * @return
     */
    public ListNode ReverseList3(ListNode root) {
        if (root == null || root.next == null) return root;

        ListNode newHead = null;
        while (root!= null){
            ListNode temp = root.next;
            root.next = newHead;
            newHead = root;
            root = temp;
        }
        return newHead;
    }

    /**
     * 2.链表内指定区间内反转    局部反转 + 切断
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween (ListNode head, int m, int n) {
        // write code here
        ListNode virtualHead = new ListNode(-1);
        virtualHead.next = head;

        ListNode pre = virtualHead;

        for (int i = 0; i < m - 1; i++) {
            pre = pre.next;
        }
        ListNode left = pre.next;

        ListNode right = pre;
        for (int i = 0; i < n - m + 1; i++) {
            right = right.next;
        }
        ListNode cur = right.next;

        //4.切断链接
        pre.next=null;
        right.next=null;

        // 局部遍历
        reverseList(left);

        pre.next = right;
        left.next = cur;

        return virtualHead.next;
    }
    public void reverseList(ListNode head){
        ListNode pre = null;
        ListNode cur = head;

        while (cur != null){
            ListNode cur_next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = cur_next;
        }
    }

    /**
     * 2.链表内指定区间内反转   一次遍历优化
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween2 (ListNode head, int m, int n) {
        // write code here
        ListNode virtualHead = new ListNode(-1);
        virtualHead.next = head;

        ListNode pre = virtualHead;

        for (int i = 0; i < m - 1; i++) {
            pre = pre.next;
        }
        
        ListNode cur = pre.next;
        ListNode Cur_next = null;
        for (int i = 0; i < n - m; i++) {
            Cur_next = cur.next;
            cur.next = Cur_next.next;
            Cur_next .next = pre.next;
            pre.next = Cur_next ;
        }
        return virtualHead.next;
    }

    /**
     * 3.链表中的节点每 k 个一组翻转  递归实现
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup (ListNode head, int k) {
        //找到每次翻转的尾部
        ListNode tail = head;
        //遍历k次到尾部
        for(int i = 0; i < k; i++){
            //如果不足k到了链表尾，直接返回，不翻转
            if(tail == null)
                return head;
            tail = tail.next;
        }
        //翻转时需要的前序和当前节点
        ListNode pre = null;
        ListNode cur = head;
        //在到达当前段尾节点前
        while(cur != tail){
            //翻转
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        //当前尾指向下一段要翻转的链表
        head.next = reverseKGroup(tail, k);
        return pre;
    }

    /**
     * 4.合并两个排序的链表
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1,ListNode list2) {
        // list1 list2为空的情况
        if(list1 == null || list2 == null){
            return list1 != null ? list1 : list2;
        }
        // 两个链表元素依次对比
        if(list1.val <= list2.val){
            // 递归计算 list1.next, list2
            list1.next = Merge(list1.next, list2);
            return list1;
        }else{
            // 递归计算 list1, list2.next
            list2.next = Merge(list1, list2.next);
            return list2;
        }
    }

    /**
     * 5.合并k个排序的链表    归并排序
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        return divide(lists, 0, lists.size() - 1);
    }

    // 合并两个链表
    // 依次顺序比较两个链表的节点值
    public ListNode Merge2(ListNode list1, ListNode list2){
        if (list1 == null){
            return list2;
        }
        if (list2 == null){
            return list1;
        }

        ListNode head = new ListNode(0);
        ListNode cur = head;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                cur.next = list1;
                list1 = list1.next;
                cur = cur.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
                cur = cur.next;
            }
        }

        if (list1 != null){
            cur.next = list1;
        }
        if (list2 != null){
            cur.next = list2;
        }

        return head.next;

    }

    // 划分区间
    // 递归调用处理每个分治后的小区间
    // 采用二分法处理每个小区间
    public ListNode divide(ArrayList<ListNode> lists, int left, int right){

        if (left > right){
            return null;
        }
        if (left == right){
            return lists.get(left);
        }

        int mid = (left + right) / 2;
        return Merge2(divide(lists, left, mid), divide(lists, mid + 1, right));
    }

    /**
     * 6.判断链表中是否有环   时间复杂度 O（n）
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        // 使用快慢指针 一个走两步，一个走一步，当重合时必定有环
        if (head == null || head.next == null){
            return false;
        }

        ListNode virHead = new ListNode(-1);
        virHead.next = head;

        ListNode slow = virHead;
        ListNode swift = virHead;

        while (swift.next != null && swift.next.next != null){
            slow = slow.next;
            swift = swift.next.next;
            if (slow == swift){
                return true;
            }
        }

        return false;
    }

    /**
     * 7.链表中环的入口节点   时间复杂度 O（n）
     * @param pHead
     * @return
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {

        if (pHead == null){
            return null;
        }

        ListNode virHead = new ListNode(-1);
        virHead.next = pHead;

        ListNode slow = virHead;
        ListNode swift = virHead;

        while (swift != null && swift.next!= null){
            slow = slow.next;
            swift = swift.next.next;
            if (slow == swift){
                break;
            }
        }
        // 当条件不满足时，一定存在环
        if (swift == null || swift.next == null){
            return null;
        }

        swift = virHead;
        while (swift != slow){
            swift = swift.next;
            slow = slow.next;
        }
        return swift;
    }

    /**
     * 8.链表中倒数最后 k 个节点   时间复杂度 O（n）
     * @param pHead
     * @param k
     * @return
     */
    public ListNode FindKthToTail (ListNode pHead, int k) {
        // write code here
        if (pHead == null) {
            return null;
        }

        ListNode temp = pHead;
        int count = 0;
        while (temp != null){
            count ++;
            temp = temp.next;
        }

        if (k  > count){
            return null;
        }

        ListNode cur = pHead;
        for (int i = 0; i < count - k; i++) {
            cur = cur.next;
        }
        return cur;
    }

    /**
     * 9.删除链表中倒数第 k 个节点   时间复杂度 O（n）
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd (ListNode head, int n) {
        // write code here
        if (head == null) {
            return null;
        }

        ListNode temp = head;
        int count = 0;
        while (temp != null){
            count ++;
            temp = temp.next;
        }

        if (n  > count){
            return null;
        }else if (n == count){
            return head.next;
        }

        ListNode cur = head;
        ListNode virHead = new ListNode(-1);
        virHead.next = head;
        ListNode pre = virHead;
        for (int i = 0; i < count - n; i++) {
            pre = pre.next;
            cur = cur.next;
        }

        pre.next = cur.next;
        return head;
    }

    /**
     * 10.两个链表的第一个公共节点   时间复杂度 O（n）
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {

        ListNode temp = pHead1;
        ListNode temp2 = pHead2;

        int count1 = 0;
        int count2 = 0;
        while (temp != null){
            count1 ++;
            temp = temp.next;
        }

        while (temp2 != null){
            count2 ++;
            temp2 = temp2.next;
        }

        int index = Math.abs(count1 - count2);

        temp = pHead1;
        temp2 = pHead2;

        if (count1 > count2){
            for (int i = 0; i < index; i++) {
                temp = temp.next;
            }
        }else if (count1 < count2){
            for (int i = 0; i < index; i++) {
                temp2 = temp2.next;
            }
        }

        while (temp != null){

            if (temp == temp2){
                break;
            }
            temp = temp.next;
            temp2 = temp2.next;
        }

        return temp;
    }

    /**
     * 11.链表相加（二）   时间复杂度 O（n） 空间复杂度 O（n）
     * @param head1 ListNode类
     * @param head2 ListNode类
     * @return ListNode类
     */
    public ListNode addInList (ListNode head1, ListNode head2) {
        // write code here
        if (head1 == null){
            return head2;
        }
        if (head2 == null){
            return head1;
        }
        // 使用栈
        Stack<ListNode> listNodeStack = new Stack<>();
        Stack<ListNode> listNodeStack2 = new Stack<>();

        ListNode temp = head1;
        ListNode temp2 = head2;

        while (temp != null){
            listNodeStack.push(temp);
            temp = temp.next;
        }

        while (temp2 != null){
            listNodeStack2.push(temp2);
            temp2 = temp2.next;
        }

        int count = Math.min(listNodeStack.size(), listNodeStack2.size());
        ListNode virHead = new ListNode(-1);
        ListNode te = virHead;
        int JinWei = 0;
        for (int i = 0; i < count; i++) {
            int addValue = listNodeStack.peek().val + listNodeStack2.peek().val;
            addValue += JinWei;
            listNodeStack.pop();
            listNodeStack2.pop();
            if (addValue >= 10){
                JinWei = 1;
                addValue %= 10;
            }else {
                JinWei = 0;
            }
            te.next = new ListNode(addValue);
            te = te.next;
        }
        if (listNodeStack.isEmpty()){
            while (!listNodeStack2.isEmpty()){
                int value = listNodeStack2.peek().val;
                value += JinWei;
                listNodeStack2.pop();
                if (value >= 10){
                    JinWei = 1;
                    value %= 10;
                }else {
                    JinWei = 0;
                }
                te.next = new ListNode(value);
                te = te.next;
            }
        }else if (listNodeStack2.isEmpty()){
            while (!listNodeStack.isEmpty()){
                int value = listNodeStack.peek().val;
                value += JinWei;
                listNodeStack.pop();
                if (value >= 10){
                    JinWei = 1;
                    value %= 10;
                }else {
                    JinWei = 0;
                }
                te.next = new ListNode(value);
                te = te.next;
            }
        }

        if (JinWei == 1){
            te.next = new ListNode(1);
        }


//        return virHead.next;
        return reverseListNode(virHead.next);
    }

    // 反转链表
    public ListNode reverseListNode(ListNode head){

        ListNode pre = null;
        ListNode cur = head;

        while (cur != null){
            ListNode curNext = cur.next;
            cur.next = pre;
            pre = cur;
            cur = curNext;
        }
        return pre;
    }

    /**
     * 12. 单链表的排序    时间复杂度 O(nlogn)
     * @param head 头节点
     * @return ListNode类
     */
    public ListNode sortInList (ListNode head) {
        // write code here
        ListNode temp = head;
        int count = 0;
        while (temp != null){
            temp = temp.next;
            count ++;
        }
        int[] ints = new int[count];

        temp = head;
        for (int i = 0; i < count; i++) {
            ints[i] = temp.val;
            temp = temp.next;
        }

        Arrays.sort(ints);

        ListNode virHead = new ListNode(-1);
        ListNode te = virHead;
        for (int i = 0; i < count; i++) {
            te.next = new ListNode(ints[i]);
            te = te.next;
        }

        return virHead.next;
    }


    /**
     * 13. 判断一个链表是否为回文结构
     * 方法一：将链表节点的值加入list
     * 方法二：使用快慢指针
     * @param head ListNode类 the head
     * @return bool布尔型
     */
    public boolean isPail (ListNode head) {
        if (head == null || head.next == null){
            return true;
        }

        // write code here
        // 快慢指针
        ListNode fast = head;
        ListNode slow = head;


        ListNode temp = head;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            temp = temp.next;
        }
        // fast 不为空，说明链表节点个数为奇数个
        if (fast != null){
            slow = slow.next;
        }

        // 反转前半部分链表
        slow = reverse13(slow);
        fast = head;

        while (slow != null){
            if (slow.val != fast.val){
                return false;
            }
            slow = slow.next;
            fast = fast.next;
        }
        return true;
    }

    public ListNode reverse13(ListNode head){

        ListNode pre = null;
        ListNode cur = head;

        while (cur != null){
            ListNode cur_next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = cur_next;
        }
        return pre;
    }

    /**
     * 14. 链表的奇偶重排
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 奇偶指针
     * @param head ListNode类
     * @return ListNode类
     */
    public ListNode oddEvenList (ListNode head) {
        // write code here
        if (head == null || head.next == null || head.next.next == null){
            return head;
        }
        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = even;
        while (even != null && even.next != null){
            odd.next = even.next;
            even.next = odd.next.next;
            odd = odd.next;
            even = even.next;
        }
        odd.next = evenHead;

        return head;
    }
    /**
     * 15. 删除有序链表中重复的元素1
     * @param head ListNode类
     * @return ListNode类
     */
    public ListNode deleteDuplicates (ListNode head) {
        // write code here
        if (head == null || head.next == null){
            return head;
        }
        ListNode virHead = new ListNode(-200);
        virHead.next = head;

        ListNode pre = virHead;
        ListNode cur = head;
        boolean flag = false;
        while (cur != null){  // 其实时间复杂度还为 O（n）
            while (pre.val == cur.val){
                if (cur.next != null){
                    cur = cur.next;
                }else {
                    flag = true;
                    pre.next = null;
                    break;
                }
            }
            if (flag){
                break;
            }
            pre.next = cur;
            pre = cur;
            cur = cur.next;
        }

        return head;
    }

    /**
     * 16. 删除有序链表中重复的元素2
     * 要求：空间复杂度 O(n), 时间复杂度：O(n)
     * 进阶：空间复杂度 O(1), 时间复杂度：O(n)
     * @param head ListNode类
     * @return ListNode类
     */
    public ListNode deleteDuplicates2 (ListNode head) {
        ListNode newhead = new ListNode(0);//设置一个初始链表
        newhead.next = head;//将head链表添加到newhead中
        ListNode pre = newhead;
        ListNode cur = head;
        int count = 0;//设置重复次数
        while(cur != null && cur.next != null){//判断条件
            if(cur.val == cur.next.val){//如果按照顺序，值相等
                cur.next = cur.next.next;//删除该元素
                count++;//将count次数加一再次进行判断
            }
            else{
                if(count > 0){
                    pre.next = cur.next;//将该元素值全部删除
                    count = 0;
                }
                else
                    pre = cur;
                cur = cur.next;//递归条件
            }
        }
        if(count > 0)//存在比如{1，2，2}，因为删除，所以上述循环条件不进行判断，在此额外进行判断
            pre.next = cur.next;
        return newhead.next;
    }

    /** 排序
     * 1. 二分查找
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 递归和迭代法
     * @param nums int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int search (int[] nums, int target) {
        // write code here
        if (nums.length == 0) return -1;

        int left = 0;
        int right = nums.length;

        while (left <= right){
            int mid = (left + right) / 2;
            if (nums[mid] > target){
                right = mid - 1;
            }else if (nums[mid] < target){
                left = mid + 1;
            }else {
                return mid;
            }
        }

        return -1;
    }

    /**
     * 2.寻找峰值
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 暴力遍历 时间复杂度为 O（n）  ->  O(logN)
     * @param nums int整型一维数组
     * @return int整型
     */
    public int findPeakElement (int[] nums) {
        // write code here
        int left = 0;
        int right = nums.length - 1;

        while (left < right){
            int mid = (left + right) / 2;
            if (nums[mid] > nums[mid + 1]){
                right = mid;
            }else {
                left = mid + 1;
            }
        }
        return right;
    }

    static final int kmod = 1000000007;
    int count = 0;
    /**
     * 3.数组中的逆序对
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 时间复杂度：O（nlogn）  空间复杂度：O（n）
     *
     *
     * @param array int整型一维数组
     * @return int整型
     */
    public int InversePairs(int [] array) {
        int[] temp = new int[array.length];
        merge_rec(array, 0, array.length - 1, temp);
        return count;
    }

    // 递归过程
    public void merge_rec(int[] array, int left, int right, int[] temp){
        if (left == right){
            return;
        }
        int mid = (left + right) / 2;
        merge_rec(array, left, mid, temp);
        merge_rec(array, mid + 1, right, temp);
        merge(array, left, mid, right, temp);
    }

    // 合并过程
    public void merge(int[] array, int left, int mid, int right, int[] temp){

        int i = left;
        int j = mid + 1;
        int k = 0;
        while (i <= mid && j <= right){
            if (array[i] < array[j]){
                temp[k++] = array[i++];
            }else {
                temp[k++] = array[j++];
//                count += 1;
                count += (mid - i + 1);
                count %= kmod;
            }
        }

        while (i <= mid){
            temp[k++] = array[i++];
        }

        while (j<=right){
            temp[k++] = array[j++];
        }

        for (int k2 = 0, t = left; t <= right; k2++, t++) {
            array[t] = temp[k2];
        }
    }

    /**
     * 利用优先队列排序的demo
     * @param arr
     * @return
     */
    public int[] PriorityQueueSort(int[] arr){

        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1.compareTo(o2);  // 从小到大
            }
        });

        for (int i = 0; i < arr.length; i++) {
            queue.add(arr[i]);
        }
        int[] newarr = new int[arr.length];

        for (int i = 0; i < arr.length; i++) {
            newarr[i] = queue.poll();
        }
        return newarr;
    }

    /**
     * 4.比较版本号
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 空间复杂度： O（1）  时间复杂度： O（n）
     * 比较版本号
     * @param version1 string字符串
     * @param version2 string字符串
     * @return int整型
     */
    public int compare (String version1, String version2) {
        int n1 = version1.length();
        int n2 = version2.length();
        int i = 0, j = 0;
        //直到某个字符串结束
        while(i < n1 || j < n2){
            long num1 = 0;
            //从下一个点前截取数字
            while(i < n1 && version1.charAt(i) != '.'){
                num1 = num1 * 10 + (version1.charAt(i) - '0');
                i++;
            }
            //跳过点
            i++;
            long num2 = 0;
            //从下一个点前截取数字
            while(j < n2 && version2.charAt(j) != '.'){
                num2 = num2 * 10 + (version2.charAt(j) - '0');
                j++;
            }
            //跳过点
            j++;
            //比较数字大小
            if(num1 > num2)
                return 1;
            if(num1 < num2)
                return -1;
        }
        //版本号相同
        return 0;
    }

    /** 二叉树
     * 1.二叉树的前序遍历、中序遍历、后序遍历
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param root TreeNode类
     * @return int整型一维数组
     */
//    public int[] preorderTraversal (TreeNode root) {
//        // write code here
//    }
      /*
        std::vector<int> preorderTraversal(TreeNode* root){
            std::vector<int> res;
            dfs(root, res);
            return res;
        }

        void dfs(TreeNode *root, std::vector<int>& res){
            if(root == NULL){
                return;
            }
            res.push_back(root->val);
            dfs(root->left, res);
            dfs(root->right, res);
        }
      */


    /**
     * 2.二叉树的层序遍历
     * @param root TreeNode类
     * @return int整型ArrayList<ArrayList<>>
     */
    public ArrayList<ArrayList<Integer>> levelOrder (TreeNode root) {
        // write code here
        if (root == null) return new ArrayList<>();

        LinkedList<TreeNode> q = new LinkedList<>();
        q.addLast(root);
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();

        while (!q.isEmpty()){
            ArrayList<Integer> temp = new ArrayList<>();
            int len = q.size();
            for (int i = 0; i < len; i++) {
                TreeNode t = q.getFirst();
                temp.add(t.val);
                q.removeFirst();
                if (t.left != null) q.addLast(t.left);
                if (t.right!= null) q.addLast(t.right);
            }
            res.add(temp);
        }
        return res;
    }

    /**
     * 3.按Z字形顺序打印二叉树
     * @param pRoot TreeNode类
     * @return int整型ArrayList<ArrayList<>>
     */
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        if (pRoot == null) return new ArrayList<>();

        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        LinkedList<TreeNode> q = new LinkedList<>();
        q.addLast(pRoot);

        int level = 0;
        while (!q.isEmpty()){
            ArrayList<Integer> temp = new ArrayList<>();
            int len = q.size();
            for (int i = 0; i < len; i++) {
                TreeNode t = q.getFirst();
                if (level % 2 == 0){
                    temp.add(t.val);
                }else {
                    temp.add(0, t.val);
                }
                q.removeFirst();
                if (t.left != null) q.addLast(t.left);
                if (t.right!= null) q.addLast(t.right);
            }
            level ++;
            res.add(temp);
        }
        return res;
    }

    /**
     * 4.二叉树的最大深度
     * 空间复杂度： O（1）  时间复杂度： O（n）
     * 可以使用递归方法求解，也可以通过层序遍历求解
     * 本题使用递归求解
     * @param root TreeNode类
     * @return int整型
     */
    public int maxDepth (TreeNode root) {
        // write code here
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }


    /**
     * 5.二叉树中和为某一值的路径1
     * 空间复杂度： O（n）  时间复杂度： O（n）
     * 进阶：空间复杂度： O（树的高度）  时间复杂度： O（n）
     * 实质为二叉树的前序遍历
     * @param root TreeNode类
     * @param sum int整型
     * @return bool布尔型
     */
    public boolean hasPathSum (TreeNode root, int sum) {
        // write code here
        if (root == null){
            return false;
        }
        return hasPathSum_dfs(root, sum);
    }

    public boolean hasPathSum_dfs(TreeNode root, int sum){
        if (root == null){ // 不是叶子节点的情况，也没找到路径
            return false;
        }

        sum -= root.val;

        if (root.left == null && root.right == null && sum == 0){ // 叶子节点且已找到路径
            return true;
        }

        return hasPathSum(root.left, sum) || hasPathSum(root.right, sum); // 是否可以剪枝？
    }

    /**
     * 6.二叉搜索树与双向链表
     * 方法一：使用中间数组  空间复杂度：O（n）  时间复杂度：O（n）
     * 方法二：原树上进行   空间复杂度： O（1）  时间复杂度： O（n）
     * 二叉搜索树的中序遍历即升序
     * @param pRootOfTree TreeNode类
     * @return bool布尔型
     */
    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) return null;
        TreeNode headList = pRootOfTree;
        while (headList.left != null){
            headList = headList.left;
        }
        Convert_dfs(pRootOfTree);
        return headList;
    }

    TreeNode preNode = null;
    public void Convert_dfs(TreeNode pRootOfTree){
        if (pRootOfTree == null){
            return;
        }
        Convert_dfs(pRootOfTree.left);

        pRootOfTree.left = preNode;

        if (preNode != null){
            preNode.right = pRootOfTree; // 存在重复指向，但是没有影响，并且必要
        }
        preNode = pRootOfTree;
        Convert_dfs(pRootOfTree.right);
    }

    /**
     * 7.对称的二叉树
     * 空间复杂度： O（n）  时间复杂度： O（n）
     * 方法一：递归  后序遍历
     * 方法二：迭代
     * @param pRoot TreeNode类
     * @return bool布尔型
     */
    boolean isSymmetrical(TreeNode pRoot) {
        return isSymmetrical_dfs(pRoot, pRoot);
    }

    public boolean isSymmetrical_dfs(TreeNode pRoot1, TreeNode pRoot2){
        if (pRoot1 == null && pRoot2 == null){
            return true;
        }

        if (pRoot1 == null || pRoot2 == null || pRoot1.val != pRoot2.val){
            return false;
        }

        return isSymmetrical_dfs(pRoot1.left, pRoot2.right) && isSymmetrical_dfs(pRoot1.right, pRoot2.left);
    }


    /**
     * 8.合并二叉树
     * 空间复杂度 o（1）  时间复杂度 O（n）
     * @param t1 TreeNode类
     * @param t2 TreeNode类
     * @return TreeNode类
     */
    public TreeNode mergeTrees (TreeNode t1, TreeNode t2) {
        // write code here
        if (t1 == null){
            return t2;
        }

        if (t2 == null){
            return t1;
        }

        TreeNode head = new TreeNode(t1.val + t2.val);
        head.left = mergeTrees(t1.left, t2.left);
        head.right = mergeTrees(t1.right, t2.right);
        System.out.println(head.val);
        return head; // 因为其他根节点没有接受，所以覆盖返回最开始的那一个
    }


    /**
     * 9.二叉树的镜像
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 空间复杂度： O（n）  时间复杂度： O（n）
     * 进阶：空间复杂度： O（1）  时间复杂度： O（n）
     *
     * @param pRoot TreeNode类
     * @return TreeNode类
     */
    public TreeNode Mirror (TreeNode pRoot) {
        // write code here
        if (pRoot == null){
            return null;
        }

        TreeNode temp = pRoot.left;
        pRoot.left = pRoot.right;
        pRoot.right = temp;

        pRoot.left = Mirror(pRoot.left);
        pRoot.right = Mirror(pRoot.right);

        return pRoot;
    }

    TreeNode preNode2 = null;
    /**
     * 10.判断是不是二叉搜索树
     * 这道题和 6.二叉树转双向链表是一样的思路
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param root TreeNode类
     * @return bool布尔型
     */
    public boolean isValidBST (TreeNode root) {
        // write code here
        return isValidBST_dfs(root);
    }

    public boolean isValidBST_dfs(TreeNode root){
        if (root == null){
            return true;
        }

        boolean l = isValidBST_dfs(root.left);

//        root.left = preNode2;
        if (preNode2!=null){
//            preNode2.right = root;
            if (preNode2.val > root.val){
                return false;
            }
        }
        preNode2 = root;

        boolean r = isValidBST_dfs(root.right);

        return l&&r;
    }

    /**
     * 11.判断是不是完全二叉树
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 根据完全二叉树的定义，采用层序遍历，当出现null时，发现null后还有节点则不是完全二叉树，反之则是
     *
     * @param root TreeNode类
     * @return bool布尔型
     */
    public boolean isCompleteTree (TreeNode root) {
        // write code here
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        boolean flag = false;
        while (!q.isEmpty()){
            TreeNode temp = q.poll();

            if (temp == null){
                flag = true;
            }else {
                if (flag){
                    return false;
                }
                q.offer(temp.left);
                q.offer(temp.right);
            }
        }
        return true;
    }


    boolean isBalanced = true;
    /**
     * 12.判断是不是平衡二叉树
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 空间复杂度 o（1）  时间复杂度 O（n）
     * 后序遍历，自底向上
     * @param root TreeNode类
     * @return bool布尔型
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        IsBalanced_Solution_dfs(root);
        return isBalanced;
    }

    public int IsBalanced_Solution_dfs(TreeNode root){
        if (root == null){
            return 0;
        }

        int l = IsBalanced_Solution_dfs(root.left);

        int r = IsBalanced_Solution_dfs(root.right);

        if (Math.abs(l - r) > 1){
            isBalanced = false;
        }

        return 1 + Math.max(l, r);
    }

    /**
     * 13.二叉树的最近公共祖先
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *  方法一：搜索路径比较 ：通过比较两个目标节点的路径，比较路径数组最后一个相等的元素便是最近公共祖先
     *  方法二：二叉树递归
     *
     * @param root TreeNode类
     * @param p int整型
     * @param q int整型
     * @return int整型
     */
    public int lowestCommonAncestor (TreeNode root, int p, int q) {
        // write code here
        // 递归实现
        lowestCommonAncestor_dfs(root, p, q);
        return target;
    }
    int target = 0;
//    boolean flag = true;
    // 前序遍历
    public void lowestCommonAncestor_dfs(TreeNode root, int p, int q){
        if (root == null){
            return;
        }

        if (root.val > p && root.val > q){
            lowestCommonAncestor(root.left, p, q);
        } else if (root.val < p && root.val < q){
            lowestCommonAncestor(root.right, p, q);
        } else {
//            if (flag){
                target = root.val;
//                flag = false;
//            }
        }
    }

    /**
     * 14.在二叉树中找到两个节点的最近公共祖先
     * 方法一：路径法 BFS（层序遍历找到节点）
     * 方法二：递归法
     *
     * @param root TreeNode类
     * @param o1 int整型
     * @param o2 int整型
     * @return int整型
     */
    public int lowestCommonAncestor2 (TreeNode root, int o1, int o2) {
        // write code here
        // 1.路径法
        return -1;
    }

    public int lowestCommonAncestor3 (TreeNode root, int o1, int o2) {
        // write code here
        // 2.递归法
        return lowestCommonAncestor3_dfs(root, o1, o2).val;
    }

    public TreeNode lowestCommonAncestor3_dfs(TreeNode root, int o1, int o2){

        if (root == null || root.val == o1 | root.val == o2){
            return root;
        }

        TreeNode left = lowestCommonAncestor3_dfs(root.left, o1, o2);
        TreeNode right = lowestCommonAncestor3_dfs(root.right, o1, o2);

        if (left == null){
            return right;
        }

        if (right == null){
            return left;
        }

        return root;
    }

    int INF = 0x3f3f3f3f;
    TreeNode emptyNode = new TreeNode(INF);
    /**
     * 15.序列化二叉树
     * 序列化可以按照前序、中序、后序
     * 使用层序序列化和反序列化
     * */
    String Serialize(TreeNode root) {
        if (root == null) return "";

        StringBuilder res = new StringBuilder();
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);


        while (!q.isEmpty()) {
            // 每次从队列中取出元素进行「拼接」，包括「正常节点」和「叶子节点对应的首位空节点」
            TreeNode poll = q.poll();
            res.append(poll.val + "_");
            // 如果取出的节点不为「占位节点」，则继续往下拓展，同时防止「占位节点」不继续往下拓展
            if (!poll.equals(emptyNode)) {
                q.offer(poll.left != null ? poll.left : emptyNode);
                q.offer(poll.right != null ? poll.right : emptyNode);
            }
        }
        return res.toString();
    }

    TreeNode Deserialize(String str) {
        if (str.equals("")) return null;

        // 根据分隔符进行分割
        String[] ss = str.split("_");
        int n = ss.length;
        // 怎么序列化就怎么反序列化
        // 使用队列进行层序遍历，起始先将 root 构建出来，并放入队列
        TreeNode root = new TreeNode(Integer.parseInt(ss[0]));
        Deque<TreeNode> d = new ArrayDeque<>();
        d.addLast(root);
        for (int i = 1; i < n - 1; i += 2) {
            TreeNode poll = d.pollFirst();
            // 每次从中取出左右节点对应 val
            int a = Integer.parseInt(ss[i]), b = Integer.parseInt(ss[i + 1]);
            // 如果左节点对应的值不是 INF，则构建「真实节点」
            if (a != INF) {
                poll.left = new TreeNode(a);
                d.addLast(poll.left);
            }
            // 如果右节点对应的值不是 INF，则构建「真实节点」
            if (b != INF) {
                poll.right = new TreeNode(b);
                d.addLast(poll.right);
            }
        }
        return root;
    }

    /**
     * 16.输出二叉树的右视图
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 求二叉树的右视图
     * 思路：先按照前序和中序遍历恢复二叉树，再按照层序遍历判断当某节点右边没有其余节点是则该节点为右边界节点
     * @return int整型一维数组
     */
    public int[] solve (int[] xianxu, int[] zhongxu) {
        // 重建二叉树
        TreeNode rebuildTree = rebuildTree(xianxu, zhongxu);
        // 层序遍历找到右边界节点
        ArrayList<ArrayList<Integer>> arrayLists = solve_levelOrder(rebuildTree);

        int[] res = new int[arrayLists.size()];
        for (int i = 0; i < arrayLists.size(); i++) {
            res[i] = arrayLists.get(i).get(arrayLists.get(i).size() - 1);
        }

        return res;
    }

    // 寻找索引
    public int findIndex(int[] arr, int target){
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target){
                return i;
            }
        }
        return -1;
    }

    // 切片
    public int[] slice(int[] arr, int start, int end){
        int[] res = new int[end - start];
        System.arraycopy(arr, start, res, 0, res.length);
        return res;
    }

    // 恢复二叉树
    public TreeNode rebuildTree(int[] pre, int[] in){
        if (pre.length == 0){
            return null;
        }

        TreeNode root = new TreeNode(pre[0]);
        int indexOfRootOnIn = findIndex(in, pre[0]);
        root.left = rebuildTree(slice(pre, 1, indexOfRootOnIn + 1), slice(in, 0, indexOfRootOnIn));
        root.right = rebuildTree(slice(pre, indexOfRootOnIn + 1, pre.length), slice(in, indexOfRootOnIn + 1, in.length));
        return root;
    }

    // 层序遍历二叉树
    public ArrayList<ArrayList<Integer>> solve_levelOrder (TreeNode root) {
        // write code here
        if (root == null) return new ArrayList<>();

        LinkedList<TreeNode> q = new LinkedList<>();
        q.addLast(root);
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();

        while (!q.isEmpty()){
            ArrayList<Integer> temp = new ArrayList<>();
            int len = q.size();
            for (int i = 0; i < len; i++) {
                TreeNode t = q.getFirst();
                temp.add(t.val);
                q.removeFirst();
                if (t.left != null) q.addLast(t.left);
                if (t.right!= null) q.addLast(t.right);
            }
            res.add(temp);
        }
        return res;
    }

    static class MyStackMin{
        Stack<Map<Integer, Integer>> stack;

        public MyStackMin() {
            this.stack = new Stack<>();
        }

        /** 堆/栈/队列
         * 1.包含 min 函数的栈
         *  一个正常栈 和 一个最小值栈   时间复杂度为 O（1） 空间复杂度为 O（n）
         *  原栈上操作（加入元组的概念）  时间复杂度为 O（1） 空间复杂度为 O（1）    x , min
         */
        public void push(int node) {
            if (stack.isEmpty()){
                HashMap<Integer, Integer> map = new HashMap<>();
                map.put(node, node);
                stack.push(map);
                return;
            }
            int peakElementMin = 0;
            for (Map.Entry<Integer, Integer> entry : stack.peek().entrySet()) {
                peakElementMin = entry.getValue();
                break;
            }

            if (node > peakElementMin){
                HashMap<Integer, Integer> map = new HashMap<>();
                map.put(node, peakElementMin);
                stack.push(map);
            }else {
                HashMap<Integer, Integer> map = new HashMap<>();
                map.put(node, node);
                stack.push(map);
            }

        }

        public void pop() {
            if (!stack.isEmpty()){
                stack.pop();
            }
        }

        public int top() {
            if (!stack.isEmpty()){
                for (Map.Entry<Integer, Integer> entry : stack.peek().entrySet()) {
                    return entry.getKey();
                }
            }
            return -20000;
        }

        public int min() {
            if (!stack.isEmpty()){
                for (Map.Entry<Integer, Integer> entry : stack.peek().entrySet()) {
                    return entry.getValue();
                }
            }
            return -20000;
        }
    }

    Stack<Character> stack_isValid;
    /**
     * 2.有效括号序列
     * 时间复杂度 O（n）  空间复杂度 O（n）
     *
     * @param s string字符串
     * @return bool布尔型
     */
    public boolean isValid (String s) {
        // write code here
        stack_isValid = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{'){
                stack_isValid.push(c);
            }else if (c == ')'){
                if (!stack_isValid.isEmpty()){
                    Character peek = stack_isValid.peek();
                    if (peek != '('){
                        return false;
                    }
                    stack_isValid.pop();
                }else {
                    return false;
                }

            }else if (c == ']'){
                if (!stack_isValid.isEmpty()){
                    Character peek = stack_isValid.peek();
                    if (peek != '['){
                        return false;
                    }
                    stack_isValid.pop();
                }else {
                    return false;
                }

            }else if (c == '}'){
                if (!stack_isValid.isEmpty()){
                    Character peek = stack_isValid.peek();
                    if (peek != '{'){
                        return false;
                    }
                    stack_isValid.pop();
                }else {
                    return false;
                }
            }
        }

        return stack_isValid.isEmpty();
    }

    /**
     * 3.滑动窗口的最大值
     * 时间复杂度 O（n）  空间复杂度 O（n）
     *
     * 考虑最大值队列
     * @param num int数组
     * @param size int
     * @return ArrayList<Integer>
     */
    public ArrayList<Integer> maxInWindows(int [] num, int size) {
        if (num.length == 0 || size > num.length || size == 0){
            return new ArrayList<>();
        }

        ArrayList<Integer> ret = new ArrayList<>();
        LinkedList<Integer> dq = new LinkedList<>();

        for (int i = 0; i < num.length; ++i) {
            while (!dq.isEmpty() && num[dq.getLast()] < num[i]) {
                dq.removeLast();
            }
            dq.addLast(i);
            // 判断队列的头部的下标是否过期
            if (dq.getFirst() + size <= i) {
                dq.removeFirst();
            }
            // 判断是否形成了窗口
            if (i + 1 >= size) {
                ret.add(num[dq.getFirst()]);
            }
        }
        return ret;
    }

    /**
     * 4.最小的k个数
     * 空间复杂度 O（n）  时间复杂度 O（nlogk）
     * 使用快排解决，空间复杂度为 O（1），平均时间复杂度为 O（n）
     *
     * @param input int数组
     * @param k int
     * @return ArrayList<Integer>
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (k==0 || k > input.length) return ret;
        int l = 0, r = input.length;
        while (l < r) {
            int p = GetLeastNumbers_Solution_quickSort(input, l, r);
            if (p+1 == k) {
                ArrayList<Integer> integers = new ArrayList<>();
                for (int i = 0; i < k; i++) {
                    integers.add(input[i]);
                }
                return integers;
            }
            if (p+1 < k) {
                l = p + 1;
            }
            else {
                r = p;
            }

        }
        return ret;
    }

    public int GetLeastNumbers_Solution_quickSort(int[] input, int l, int r){

        int pivot = input[r-1]; // 最后一个值作为分割点
        int i = l;
        for (int j=l; j<r-1; ++j) {
            if (input[j] < pivot) {
                int temp = input[j];
                input[j] = input[i];
                input[i] = temp;
                i++;
            }
        }
        int temp = input[i];
        input[i] = input[r-1];
        input[r-1] = temp;
        return i;
    }

    /**
     * 5.寻找第k大的元素
     * 空间复杂度 O（1）  时间复杂度 O（nlogn）
     * 使用优先队列
     *
     * @param a int数组
     * @param n int
     * @param K int
     * @return int
     */
    public int findKth(int[] a, int n, int K) {
        // write code here
        if (n == 0 || K > n) return -1;
        PriorityQueue<Integer> q = new PriorityQueue<>();
        for (int i = 0; i < K; i++) {
            q.add(a[i]);
        }

        for (int i = K; i < n; i++) {
            if (q.peek() < a[i]){
                q.poll();
                q.add(a[i]);
            }

        }

        return q.peek();
    }

    //小顶堆，元素数值都比大顶堆大
    private PriorityQueue<Integer> max = new PriorityQueue<>();
    //大顶堆，元素数值较小
    private PriorityQueue<Integer> min = new PriorityQueue<>((o1, o2)->o2.compareTo(o1));
    //维护两个堆，取两个堆顶部即与中位数相关
    /**
     * 6.数据流中的中位数
     * 空间复杂度 O（n）  时间复杂度 O（nlogn）
     *
     *
     * @param num int数组
     */
    public void Insert(Integer num) {
        //先加入较小部分
        min.offer(num);
        //将较小部分的最大值取出，送入到较大部分
        max.offer(min.poll());
        //平衡两个堆的数量
        if(min.size() < max.size())
            min.offer(max.poll());
    }

    public Double GetMedian() {
        //奇数个
        if(min.size() > max.size())
            return (double)min.peek();
        else
            //偶数个
            return (double)(min.peek() + max.peek()) / 2;
    }


    // 使用 map 维护一个运算符优先级（其中加减法优先级相同，乘法有着更高的优先级）
    Map<Character, Integer> map = new HashMap<Character, Integer>(){{
        put('-', 1);
        put('+', 1);
        put('*', 2);
    }};
    /**
     * 7.表达式求值
     *
     * 空间复杂度 O（n）时间复杂度 O（n）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回表达式的值
     * @param s string字符串 待计算的表达式
     * @return int整型
     */
    public int solve(String s) {
        // 将所有的空格去掉
        s = s.replaceAll(" ", "");

        char[] cs = s.toCharArray();
        int n = s.length();

        // 存放所有的数字
        Deque<Integer> nums = new ArrayDeque<>();
        // 为了防止第一个数为负数，先往 nums 加个 0
        nums.addLast(0);
        // 存放所有「非数字以外」的操作
        Deque<Character> ops = new ArrayDeque<>();

        for (int i = 0; i < n; i++) {
            char c = cs[i];
            if (c == '(') {
                ops.addLast(c);
            } else if (c == ')') {
                // 计算到最近一个左括号为止
                while (!ops.isEmpty()) {
                    if (ops.peekLast() != '(') {
                        calc(nums, ops);
                    } else {
                        ops.pollLast();
                        break;
                    }
                }
            } else {
                if (isNumber(c)) {
                    int u = 0;
                    int j = i;
                    // 将从 i 位置开始后面的连续数字整体取出，加入 nums
                    while (j < n && isNumber(cs[j])) u = u * 10 + (cs[j++] - '0');
                    nums.addLast(u);
                    i = j - 1;
                } else {
                    if (i > 0 && (cs[i - 1] == '(' || cs[i - 1] == '+' || cs[i - 1] == '-')) {
                        nums.addLast(0);
                    }
                    // 有一个新操作要入栈时，先把栈内可以算的都算了
                    // 只有满足「栈内运算符」比「当前运算符」优先级高/同等，才进行运算
                    while (!ops.isEmpty() && ops.peekLast() != '(') {
                        char prev = ops.peekLast();
                        if (map.get(prev) >= map.get(c)) {
                            calc(nums, ops);
                        } else {
                            break;
                        }
                    }
                    ops.addLast(c);
                }
            }
        }
        // 将剩余的计算完
        while (!ops.isEmpty() && ops.peekLast() != '(') calc(nums, ops);
        return nums.peekLast();
    }
    // 计算逻辑：从 nums 中取出两个操作数，从 ops 中取出运算符，然后根据运算符进行计算即可
    void calc(Deque<Integer> nums, Deque<Character> ops) {
        if (nums.isEmpty() || nums.size() < 2) return;
        if (ops.isEmpty()) return;
        int b = nums.pollLast(), a = nums.pollLast();
        char op = ops.pollLast();
        int ans = 0;
        if (op == '+') ans = a + b;
        else if (op == '-') ans = a - b;
        else if (op == '*') ans = a * b;
        nums.addLast(ans);
    }
    boolean isNumber(char c) {
        return Character.isDigit(c);
    }


    /** 哈希
     * 1.两数之和
     *
     * 空间复杂度 O（n）   时间复杂度 O（nlogn）   使用 hash 时间复杂度为 O（n）  哈希表查找的时间复杂度为 O（1）
     * @param numbers int整型一维数组
     * @param target int整型
     * @return int整型一维数组
     */
    public int[] twoSum (int[] numbers, int target) {
        // write code here
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])){
                return new int[]{map.get(target - numbers[i]) + 1, i+1};
            }else {
                map.put(numbers[i], i);
            }
        }

        throw new IllegalArgumentException("No solution");
    }

    /**
     * 2.数组中出现次数超过一半的数字
     *
     * 空间复杂度为 O（1） 时间复杂度为 O（n）
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int [] array) {
        if (array.length == 1) return array[0];
        int targetLen = array.length / 2;

        HashMap<Integer, Integer> map = new HashMap<>();

        for (int j : array) {
            if (map.containsKey(j)) {
                int newValue = map.get(j) + 1;
                map.put(j, newValue);
                if (newValue > targetLen) {
                    return j;
                }
            } else {
                map.put(j, 1);
            }
        }

        return -1;
    }


    /**
     * 3.数组中只出现一次的两个数字
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 空间复杂度为 O（1） 时间复杂度为 O（n）
     * @param array int整型一维数组
     * @return int整型一维数组
     */
    public int[] FindNumsAppearOnce (int[] array) {
        // write code here

        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i : array) {
            if (map.containsKey(i)) {
                int newValue = map.get(i) + 1;
                map.put(i, newValue);
            } else {
                map.put(i, 1);
            }
        }

        int[] res = new int[2];
        int c = 0;
        for (Map.Entry<Integer, Integer> integerIntegerEntry : map.entrySet()) {
            if (integerIntegerEntry.getValue() == 1){
                res[c++] = integerIntegerEntry.getKey();
            };
        }

        return res;
    }

    /**
     * 4.缺失的第一个正整数
     *
     * 空间复杂度为 O（1） 时间复杂度为 O（n）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int minNumberDisappeared (int[] nums) {
        // write code here
        int n = nums.length;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, 1);
        }
        int c = 1;
        while (map.containsKey(c)){
            c++;
        }

        return c;
    }

    /**
     * 5.三数之和
     *
     * 空间复杂度为 O（n2） 时间复杂度为 O（n2）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 方法一：双指针
     * 方法二：回溯
     * @param num int整型一维数组
     * @return ArrayList<ArrayList<Integer>>
     */
    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        if (num.length < 3) return new ArrayList<>();
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();

        Arrays.sort(num);
        for (int i = 0; i < num.length; i++) {
            if(i>0 && num[i]==num[i-1])continue;
            int j = i+1;
            int k = num.length - 1;
            while (j < k){
                if (num[i] + num[j] + num[k] > 0){
                    k --;
                }else if (num[i] + num[j] + num[k] < 0){
                    j ++;
                }else {
                    ArrayList<Integer> temp = new ArrayList<>();
                    temp.add(num[i]);
                    temp.add(num[j]);
                    temp.add(num[k]);
                    res.add(temp);
                    while (num[j] == num[j+1] && j+1 < k){
                        j++;
                    }
                    while (num[k-1] == num[k] && k-1>j){
                        k--;
                    }
                    j++;
                    k--;
                }
            }

        }

        return res;
    }

    /** 递归/回溯
     * 1.没有重复项数组的全排列
     *
     * 空间复杂度为 O（n！） 时间复杂度为 O（n！）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param num int整型一维数组
     * @return ArrayList<ArrayList<Integer>>
     */
    public ArrayList<ArrayList<Integer>> permute(int[] num) {
        if (num.length == 0) return new ArrayList<>();
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();

        boolean[] used = new boolean[num.length];
        permute_dfs(res, temp, num);

        return res;
    }

    public void permute_dfs(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> temp,int[] num){
        if (temp.size() == num.length){
            res.add(new ArrayList<>(temp));
            return;
        }

        for (int j : num) {
            if (temp.contains(j)) continue;
            temp.add(j);
            permute_dfs(res, temp, num);
            temp.remove(temp.size() - 1);
        }
    }


    /**
     * 2.有重复项数字的全排列
     *
     * 空间复杂度为 O（n！） 时间复杂度为 O（n！）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param num int整型一维数组
     * @return ArrayList<ArrayList<Integer>>
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
        if (num.length == 0) return new ArrayList<>();
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        Arrays.sort(num);

        boolean[] used = new boolean[num.length];
        permuteUnique_dfs(res, temp, num, used);

        return res;


    }

    public void permuteUnique_dfs(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> temp,int[] num, boolean[] used) {
        if (temp.size() == num.length){
            res.add(new ArrayList<>(temp));
            return;
        }

        for (int j = 0; j < num.length; j++) {
            if (used[j]) continue;
            if (j > 0 && num[j] == num[j-1] && !used[j-1]){
                continue;
            }

            temp.add(num[j]);
            used[j] = true;
            permuteUnique_dfs(res, temp, num, used);
            temp.remove(temp.size() - 1);
            used[j] = false;
        }
    }

    /**
     * 3.判断岛屿数量
     *
     * dfs 和 bfs解法
     * 1代表陆地，0代表海洋，考虑相邻上下左右
     * @param grid char字符型二维数组
     * @return int整型
     */
    public int solve (char[][] grid) {
        // write code here
        // dfs
        if (grid.length == 0) return 0;

        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1'){
                    count++;
                    solve_dfs(grid, i, j);
                }
            }
        }
        return count;
    }
    public void solve_dfs(char[][] grid, int i, int j){

        grid[i][j] = '0'; // 当前置为 0

        if (i - 1 >= 0 && grid[i - 1][j] == '1'){
            solve_dfs(grid, i - 1, j);
        }
        if (i + 1 < grid.length && grid[i + 1][j] == '1'){
            solve_dfs(grid, i + 1, j);
        }
        if (j - 1 >= 0 && grid[i][j - 1] == '1'){
            solve_dfs(grid, i, j - 1);
        }
        if (j + 1 < grid[0].length && grid[i][j + 1] == '1'){
            solve_dfs(grid, i, j + 1);
        }

    }

    public int solve2 (char[][] grid) {
        // write code here
        // bfs
        if (grid.length == 0) return 0;
        int count = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {

                Queue<Integer> queue1 = new ArrayDeque<>(); // 队列存放二维坐标  i
                Queue<Integer> queue2 = new ArrayDeque<>(); // 队列存放二维坐标  j

                if (grid[i][j] == '1') {
                    queue1.add(i);
                    queue2.add(j);

                    while (!queue1.isEmpty() && !queue2.isEmpty()){

                        int x = queue1.poll();
                        int y = queue2.poll();

                        grid[x][y] = '0';

                        if (x - 1 >= 0 && grid[x - 1][y] == '1'){
                            queue1.add(x - 1);
                            queue2.add(y);
                            grid[x - 1][y] = '0';
                        }
                        if (x + 1 < grid.length && grid[x + 1][y] == '1'){
                            queue1.add(x + 1);
                            queue2.add(y);
                            grid[x + 1][y] = '0';
                        }
                        if (y - 1 >= 0 && grid[x][y - 1] == '1'){
                            queue1.add(x);
                            queue2.add(y - 1);
                            grid[x][y - 1] = '0';
                        }
                        if (y + 1 < grid[0].length && grid[x][y + 1] == '1'){
                            queue1.add(x);
                            queue2.add(y + 1);
                            grid[x][y + 1] = '0';
                        }
                    }
                    count ++;
                    }
                }
            }
        return count;
    }

    /**
     * 4.字符串的排列
     *
     * 空间复杂度为 O（n！）
     * 时间复杂度为 O（n！）
     * @param str
     * @return ArrayList<String> 数组
     */
    public ArrayList<String> Permutation(String str) {
        if (str.length() == 0) return new ArrayList<>();
        ArrayList<String> res = new ArrayList<>();

        boolean[] used = new boolean[str.length()];
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        Permutation_backtrack(chars, res, new StringBuilder(), used);

        return res;
    }


    public void Permutation_backtrack(char[] str, ArrayList<String> res, StringBuilder temp, boolean[] used){
        if (temp.length() == str.length){
            res.add(temp.toString());
            return;
        }

        for (int i = 0; i < str.length; i++) {
            if (used[i]){
                continue;
            }

            if(i > 0 && str[i - 1] == str[i] && !used[i - 1]) // 剪枝很重要，不然会超时
                //当前的元素str[i]与同一层的前一个元素str[i-1]相同且str[i-1]已经用过了
                continue;

            temp.append(str[i]);
            used[i] = true;
            Permutation_backtrack(str, res, temp, used);
            temp.deleteCharAt(temp.length() - 1);
            used[i] = false;
        }
    }

    Set<Integer> column = new HashSet<Integer>(); //标记列不可用
    Set<Integer> posSlant = new HashSet<Integer>();//标记正斜线不可用
    Set<Integer> conSlant = new HashSet<Integer>();//标记反斜线不可用
    int result = 0;
    /**
     * 5.n皇后问题
     * 空间复杂度 O（1）
     * 时间复杂度 O（n！）
     * @param n int整型 the n
     * @return int整型
     */
    public int Nqueen (int n) {
        // write code here
        compute(0, n); // 0 代表皇后计数， n代表皇后总数
        return result;
    }
    private void compute(int i, int n){
        if(i == n){ // 皇后计数满足皇后总数，则存在一种排列方案
            result++;
            return;
        }
        for(int j = 0; j < n; j++){
            if(column.contains(j) || posSlant.contains(i - j) || conSlant.contains(i + j)){
                continue;
            }
            column.add(j);//列号j
            posSlant.add(i - j);//行号i - 列号j 正斜线
            conSlant.add(i + j);//行号i + 列号j 反斜线
            compute(i + 1, n); //计算下一行
            column.remove(j); //完成上一步递归计算后，清除
            posSlant.remove(i - j);
            conSlant.remove(i + j);
        }
    }

    /**
     * 6.括号生成
     * 空间复杂度 O（n）
     * 时间复杂度 O（2^n）
     *
     * @param n int整型
     * @return string字符串ArrayList
     */
    public ArrayList<String> generateParenthesis (int n) {
        // write code here
        if (n == 0) return new ArrayList<>();

        ArrayList<String> res = new ArrayList<>();

        generateParenthesis_backTrack(res, new StringBuilder(), 0, 0, n);

        return res;
    }

    public void generateParenthesis_backTrack(ArrayList<String> res, StringBuilder temp, int leftCount, int rightCount, int n){
        if (leftCount == n && rightCount == n){
            res.add(temp.toString());
            return;
        }

        if (leftCount < n){
            temp.append("(");
            generateParenthesis_backTrack(res, temp, leftCount + 1, rightCount, n);
            temp.deleteCharAt(temp.length() - 1);
        }
        if (rightCount < leftCount && rightCount < n){
            temp.append(")");
            generateParenthesis_backTrack(res, temp, leftCount, rightCount + 1, n);
            temp.deleteCharAt(temp.length() - 1);
        }
    }

    /**
     * 7.矩阵最长递增路径
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 递增路径的最大长度
     * @param matrix int整型二维数组 描述矩阵的每个数
     * @return int整型
     */
    //记录四个方向
    private int[][] dirs = new int[][] {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // 分别对应 上下
    private int n, m;
    //深度优先搜索，返回最大单元格数
    public int dfs(int[][] matrix, int[][] dp, int i, int j) {
        if(dp[i][j] != 0)
            return dp[i][j];
        dp[i][j]++;
        for (int k = 0; k < 4; k++) {
            int nexti = i + dirs[k][0];
            int nextj = j + dirs[k][1];
            //判断条件
            if(nexti >= 0 && nexti < n && nextj >= 0 && nextj < m && matrix[nexti][nextj] > matrix[i][j]) // 没有越界，并且递增
                dp[i][j] = Math.max(dp[i][j], dfs(matrix, dp, nexti, nextj) + 1);
        }
        return dp[i][j];
    }
    public int solve (int[][] matrix) {
        //矩阵不为空
        if (matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int res = 0;
        n = matrix.length;
        m = matrix[0].length;
        //i，j处的单元格拥有的最长递增路径
        int[][] dp = new int[m + 1][n + 1];
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                //更新最大值
                res = Math.max(res, dfs(matrix, dp, i, j));
        return res;
    }


    /** 动态规划
     * 1.跳台阶
     *
     * 时间复杂度 O（n）  空间复杂度 O（1）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param target 整数
     * @return int
     */
    public int jumpFloor(int target) {
        int[] dp = new int[target + 1];

        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= target; i++) {
            dp[i] = dp[i - 2] + dp[i - 1];
        }

        return dp[target];
    }

    /**
     * 2.最小花费爬楼梯
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param cost int整型一维数组
     * @return int整型
     */
    public int minCostClimbingStairs (int[] cost) {
        // write code here
        int[] dp = new int[cost.length + 1];

        for (int i = 2; i <= cost.length; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]); // dp[2] 是 索引下标为 0 和 1花费的最小值
        }

        return dp[cost.length];
    }


    String res = "";
    /**
     * 3.最长公共子序列2
     *
     * 时间复杂度 O(n^2) 空间复杂度 O(n^2)
     * longest common subsequence
     * @param s1 string字符串 the string
     * @param s2 string字符串 the string
     * @return string字符串
     */
    public String LCS (String s1, String s2) {
        // write code here
        if (s1.length() == 0 || s2.length() == 0) return "-1";

        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        int[][] record = new int[s1.length() + 1][s2.length() + 1]; // 1表示左上方、2表示左方、3表示上方

        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    record[i][j] = 1; // 记录左上方
                }else {
                    if (dp[i - 1][j] > dp[i][j - 1]){
                        dp[i][j] = dp[i - 1][j];
                        record[i][j] = 3;
                    }else {
                        dp[i][j] = dp[i][j - 1];
                        record[i][j] = 2;
                    }
                }
            }
        }

        // 根据 record 拼接 字符串
        ans(s1.length(), s2.length(), record, s1, s2);

        if(!res.isEmpty()){
            return res;
        }else {
            return "-1";
        }
    }

    public void ans(int i, int j, int[][] record, String s1, String s2){
        if (i == 0 || j == 0){
            return;
        }

        if (record[i][j] == 1){
            ans(i - 1, j - 1, record, s1, s2);
            res += s1.charAt(i - 1);

        }else {
            if (record[i][j] == 2){
                ans(i, j - 1, record, s1, s2);
            }else {
                ans(i - 1, j, record, s1, s2);

            }
        }
    }


    /**
     * 4.最长公共子串
     *
     * 时间复杂度 O(n^2) 空间复杂度 O(n^2)
     * longest common substring
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    public String LCS2 (String str1, String str2) {
        // write code here
        int[][] dp = new int[str1.length() + 1][str2.length() + 1];
        int maxLen = 0;
        int maxLastIndex = 0;

        for (int i = 0; i < str1.length(); i++) {
            for (int j = 0; j < str2.length(); j++) {
                if (str1.charAt(i) == str2.charAt(j)){
                    dp[i + 1][j + 1] = dp[i][j] + 1;

                    if (dp[i + 1][j + 1] > maxLen) {
                        maxLen = dp[i + 1][j+1];
                        maxLastIndex = i;
                    }
//                    maxLen = Math.max(dp[i + 1][j + 1], maxLen);
//                    maxLastIndex = i;
                }else {
                    dp[i + 1][j + 1] = 0;  // 注意这里和最长公共子序列的区别
                }
            }
        }

        return str1.substring(maxLastIndex - maxLen + 1, maxLastIndex + 1);
    }

    /**
     * 5.不同路径的数目1
     *
     * 时间复杂度 O（mn）  空间复杂度 O（mn）
     * 进阶 空间复杂度 O（1）  时间复杂度 O（min（m,n））
     * @param m int整型
     * @param n int整型
     * @return int整型
     */
    public int uniquePaths (int m, int n) {
        // write code here
        if (m == 1 || n == 1) return 1;

        int[][] dp = new int[m + 1][n + 1];

        //第一行初始化，只有一条路径
        for(int i=0;i<n;i++){
            dp[0][i] = 1;
        }
        //第一列初始化，只有一条路径
        for(int i=0;i<m;i++){
            dp[i][0] =1;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }

    int res2 = 0; // 使用回溯 会超时
    public int uniquePaths2 (int m, int n) {
        // write code here
        uniquePaths2_dfs(m, n, 0 , 0);
        return res2;
    }

    public void uniquePaths2_dfs(int m, int n, int i, int j){
        if (i < 0 || i >= m || j < 0 || j >= n){
            return;
        }
        if ((i == m - 1) && (j == n - 1)){
            res2 += 1;
        }

        uniquePaths2_dfs(m, n, i, j+1);
        uniquePaths2_dfs(m, n, i + 1, j);

    }


    /**
     * 6.矩阵的最小路径和
     * @param matrix int整型二维数组 the matrix
     * @return int整型
     */
    public int minPathSum (int[][] matrix) {
        // write code here
        int[][] dp = new int[matrix.length][matrix[0].length];

        dp[0][0] = matrix[0][0];

        for (int i = 1; i < matrix.length; i++) {
            dp[i][0] = dp[i - 1][0] + matrix[i][0];
        }

        for (int i = 1; i < matrix[0].length; i++) {
            dp[0][i] = dp[0][i - 1] + matrix[0][i];
        }

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j];
            }
        }

        return dp[matrix.length - 1][matrix[0].length - 1];
    }


    /**
     * 7.把数字翻译成字符串
     * 解码
     *
     * 时间复杂度 O（n）  空间复杂度 O（n）
     * 'a' -> 1,  'b' -> 2,  'z' -> 26
     * @param nums string字符串 数字串
     * @return int整型
     */
    public int solve2 (String nums) {
        // write code here
        if (nums.equals("0")) return 0;
        if (nums.equals("10") || nums.equals("20")) return 1;
        if (nums.length() == 1) return 1;


        // 当0的前面不是 1或者2时无法译码
        for (int i = 1; i < nums.length(); i++) {
            if (nums.charAt(i) == '0'){
                if (nums.charAt(i - 1) != '1' && nums.charAt(i - 1) != '2'){
                    return 0;
                }
            }
        }

        int[] dp = new int[nums.length() + 1];

        dp[0] = 1;

        if (((nums.charAt(0) - '0') * 10 + (nums.charAt(1) - '0') >= 11 && (nums.charAt(0) - '0') * 10 + (nums.charAt(1) - '0') <= 19)
                || ((nums.charAt(0) - '0') * 10 + (nums.charAt(1) - '0') >= 21 && (nums.charAt(0) - '0') * 10 + (nums.charAt(1) - '0') <= 26)) {
            // 存在两中译码方案
            dp[1] = 2;
        }else {
            dp[1] = 1;
        }

        for (int i = 2; i < nums.length(); i++) {
            if (((nums.charAt(i-1) - '0') * 10 + (nums.charAt(i) - '0') >= 11 && (nums.charAt(i-1) - '0') * 10 + (nums.charAt(i) - '0') <= 19)
            || ((nums.charAt(i-1) - '0') * 10 + (nums.charAt(i) - '0') >= 21 && (nums.charAt(i-1) - '0') * 10 + (nums.charAt(i) - '0') <= 26)){
                // 存在两中译码方案
                dp[i] = dp[i - 1] + dp[i - 2];
            }else {
                // 只有一种译码方案
                dp[i] = dp[i - 1];
            }
        }

        return dp[nums.length() - 1];
    }

    /**
     * 8.兑换零钱1
     *
     * 最少货币数
     *
     * @param arr int整型一维数组 the array
     * @param aim int整型 the target
     * @return int整型
     */
    public int minMoney (int[] arr, int aim) {
        // write code here
        if (aim < 1) return 0;
        int[] dp = new int[aim + 1];  //dp[i] 表示要凑出 i 元钱需要的最小货币数

        Arrays.fill(dp, aim + 1); // 填充为最大货币数
        dp[0] = 0;

        for (int i = 1; i <= aim; i++) { // 遍历容量
            for (int k : arr) { // 遍历货币
                if (k <= i) {
                    dp[i] = Math.min(dp[i], dp[i - k] + 1);
                }
            }
        }
        return dp[aim] > aim?-1:dp[aim];
    }

    /**
     * 9.最长上升子序列1
     *
     * 时间复杂度 O（n^2）  空间复杂度 O（n）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 给定数组的最长严格上升子序列的长度。
     * @param arr int整型一维数组 给定的数组
     * @return int整型
     */
    public int LIS (int[] arr) {
        // write code here
        int n=arr.length;
        //特殊请款判断
        if(n==0) return 0;
        //dp[i]表示以下标i结尾的最长上升子序列长度
        int[] dp=new int[n];
        //初始化为1
        Arrays.fill(dp, 1);
        int res = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<i;j++){
                if(arr[i]>arr[j]){
                    //只要前面某个数小于当前数，则要么长度在之前基础上加1，要么保持不变，取较大者
                    dp[i]=Math.max(dp[i],dp[j]+1); // 以 i 索引结尾的最长上升子序列的最大值
                }
            }
            res = Math.max(dp[i], res); // 更新最大值
        }
        //返回所有可能中的最大值
        return res;
    }

    /**
     * 10.连续子数组的最大值
     *
     * 时间复杂度 O（n）  空间复杂度 O（n）
     * 进阶：时间复杂度 O（n）  空间复杂度 O（1）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 给定数组的最长严格上升子序列的长度。
     * @param array int整型一维数组 给定的数组
     * @return int整型
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array.length == 1) return array[0];

        int[] dp = new int[array.length]; // dp 代表以 i 结尾子串和的最大值
        dp[0] = array[0];
        int res = 0;

        for (int i = 1; i < array.length; i++) {
            dp[i] = Math.max(array[i], dp[i - 1] + array[i]);
            res = Math.max(res, dp[i]); // 因为并不是最后一个为最大值
        }

        return dp[array.length - 1];
    }

    // 进阶
    public int FindGreatestSumOfSubArray2(int[] array) {
        int sum = 0;
        int max = array[0];
        for (int j : array) {
            // 优化动态规划，确定sum的最大值
            sum = Math.max(sum + j, j);
            // 每次比较，保存出现的最大值
            max = Math.max(max, sum);
        }
        return max;
    }


    /**
     * 11.最长回文字串
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param A string字符串
     * @return int整型
     */
    public int getLongestPalindrome (String A) {
        // write code here
        if (A.length() == 0) return 0;

        boolean[][] dp = new boolean[A.length()][A.length()]; // dp 表示 前一个索引 到 后一个索引是否构成回文串
        int res = 0;

        for (int c = 0; c <= A.length(); c++) {  // 遍历字串长度 c 为字串长度
            for (int left = 0; left + c < A.length(); left++) { // left 左边界
                int right = left + c; // right 右边界
                if (A.charAt(left) == A.charAt(right)){
                    if (c <= 1){
                        dp[left][right] = true;
                    }else {
                        dp[left][right] = dp[left + 1][right - 1];
                    }

                    if (dp[left][right]){
                        res = c + 1;
                    }

                }
            }
        }
        return res;
    }

    /**
     * 12.数字字符串转换成 IP 地址
     *
     * 回溯 + 剪枝
     * 空间复杂度 O（n！）  时间复杂度 O（n！）
     * @param s string字符串
     * @return string字符串ArrayList
     */
    //记录分段IP数字字符串
    private String nums = "";
    //step表示第几个数字，index表示字符串下标
    public void dfs(String s, ArrayList<String> res, int step, int index){
        //当前分割出的字符串
        String cur = "";
        //分割出了四个数字
        if(step == 4){
            //下标必须走到末尾
            if(index != s.length())
                return;
            res.add(nums);
        }else{
            //最长遍历3位
            for(int i = index; i < index + 3 && i < s.length(); i++){
                cur += s.charAt(i);
                //转数字比较
                int num = Integer.parseInt(cur);
                String temp = nums;
                //不能超过255且不能有前导0
                if(num <= 255 && (cur.length() == 1 || cur.charAt(0) != '0')){
                    //添加点
                    if(step - 3 != 0) // 最后一个不能加 .
                        nums += cur + ".";
                    else
                        nums += cur;
                    //递归查找下一个数字
                    dfs(s, res, step + 1, i + 1);
                    //回溯
                    nums = temp;
                }
            }
        }
    }
    public ArrayList<String> restoreIpAddresses(String s) {
        ArrayList<String> res = new ArrayList<String>();
        dfs(s, res, 0, 0);
        return res;
    }


    /**
     * 13. 打家劫舍1
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int rob (int[] nums) {
        // write code here
        // dp[i]表示长度为i的数组，最多能偷取多少钱
        int[] dp = new int[nums.length + 1];
        //长度为1只能偷第一家
        dp[1] = nums[0];
        for(int i = 2; i <= nums.length; i++)
            //对于每家可以选择偷或者不偷
            dp[i] = Math.max(dp[i - 1], nums[i - 1] + dp[i - 2]); // 注意这里 nums[i - 1] 因为 i - 1 就是当前家
        return dp[nums.length];
    }

    /**
     * 14. 打家劫舍2
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int rob2 (int[] nums) {
        // write code here
        int[] dp = new int[nums.length + 1];
        dp[1] = nums[0];
        // 最后一家不偷
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        int res = dp[nums.length];
        Arrays.fill(dp, 0);
        // 第一家不偷
        dp[1] = 0;
        for (int i = 2; i <= nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }

        return Math.max(res, dp[nums.length]);
    }


    /**
     * 15. 买卖股票的最好时机1
     *
     * @param prices int整型一维数组
     * @return int整型
     */
    public int maxProfit (int[] prices) {
        // write code here
        if (prices.length < 2) return 0;

        int[][] dp = new int[prices.length][2]; // dp[i][0]表示第i天不持股，dp[i][1]表示第i天持股
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);

        }

        return Math.max(dp[prices.length - 1][0], dp[prices.length - 1][1]);
    }

    /**
     * 16. 买卖股票的最好时机2
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算最大收益
     * @param prices int整型一维数组 股票每一天的价格
     * @return int整型
     */
    public int maxProfit2 (int[] prices) {
        // write code here
        if (prices.length < 2) return 0;

        int[][] dp = new int[prices.length][2]; // dp[i][0]表示第i天不持股，dp[i][1]表示第i天持股
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] -prices[i]);

        }

        return Math.max(dp[prices.length - 1][0], dp[prices.length - 1][1]);
    }

    /**
     * 17. 买卖股票的最好时机3
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 两次交易所能获得的最大收益
     * @param prices int整型一维数组 股票每一天的价格
     * @return int整型
     */
    public int maxProfit3 (int[] prices) {
        // write code here
        int[][] dp = new int[prices.length][5];
        // dp[i][0] 表示第i天为止还没购买股票
        // dp[i][1] 表示第i天为止购买一次股票，还没卖出的情况
        // dp[i][2] 表示第i天为止购买一个股票且卖出一次股票的情况
        // dp[i][3] 表示第i天为止购买两次股票且卖出一次股票的情况
        // dp[i][4] 表示第i天为止购买两次股票且卖出两次股票的情况
        Arrays.fill(dp[0], -10000);

        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = Math.max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
            dp[i][2] = Math.max(dp[i - 1][1] + prices[i], dp[i - 1][2]);
            dp[i][3] = Math.max(dp[i - 1][2] - prices[i], dp[i - 1][3]);
            dp[i][4] = Math.max(dp[i - 1][3] + prices[i], dp[i - 1][4]);
        }

        return Math.max(dp[prices.length - 1][2], Math.max(0, dp[prices.length - 1][4]));
    }

    /**
     * 18. 编辑距离1
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 对字符串的三种操作
     * 插入、删除、修改
     *
     * @param str1 string字符串
     * @param str2 string字符串
     * @return int整型
     */
    public int editDistance (String str1, String str2) {
        // write code here
        int[][] dp = new int[str1.length() + 1][str2.length() + 1]; // dp 表示第 第一个字符串首部到 第 i 位的字串 修改到 第二个字符串首部到 第 j 位的字符需要修改的距离

        // 初始化边界
        for (int i = 1; i <= str1.length(); i++) {
            dp[i][0] = dp[i - 1][0] + 1;
        }
        for (int i = 1; i <= str2.length(); i++) {
            dp[0][i] = dp[0][i - 1] + 1;
        }

        for (int i = 1; i <= str1.length(); i++) {
            for (int j = 1; j <= str2.length(); j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }

        return dp[str1.length()][str2.length()];
    }


    /** 字符串
     * 1. 字符串变形
     *
     * 时间复杂度 O（n）  空间复杂度 O（n）
     *
     * @param s string字符串
     * @param n int整型
     * @return string字符串
     */
    public String trans(String s, int n) {
        // write code here
        if (s.length() == 0 || n == 0) return "";
        char[] chars = s.toCharArray();

        for (int i = 0; i < n; i++) {
            if (chars[i] == ' '){
                continue;
            }
            if (chars[i] >= 'a' && chars[i] <= 'z'){
                String temp = chars[i] + "";
                chars[i] = temp.toUpperCase().charAt(0);
            }else {
                String temp = chars[i] + "";
                chars[i] = temp.toLowerCase().charAt(0);
            }
        }

        System.out.println(chars);

        StringBuilder res1 = new StringBuilder();

        String t = "";
        for (char c:chars){
            if (c == ' '){
                res1.append(t);
                res1.append(' ');
                t = "";
            }else {
                t+=c;
            }
        }
        res1.append(t);
        System.out.println(res1);

        StringBuilder r = new StringBuilder();
        String[] s1 = res1.toString().split(" ");

        if (s1.length == 0) return s;

        for (int i = s1.length - 1; i >=0 ; i--) {
            r.append(s1[i]);
            r.append(" ");
        }
        r.delete(r.length()-1, r.length());

        System.out.println(r);

        if (res1.length() > r.length()){
            r.insert(0, res1.substring(r.length(), res1.length()));
        }

        return r.toString();
    }

    /**
     * 2.最长公共前缀
     *
     * 时间复杂度 O（n*len）  空间复杂度 O（1）
     * @param strs string字符串一维数组
     * @return string字符串
     */
    public String longestCommonPrefix (String[] strs) {
        // write code here
        if (strs.length == 0) return "";
        if (strs.length == 1) return strs[0];
        int minLenIndex = 0;
        for (int i = 1; i < strs.length; i++) {
            if (strs[i].length() < strs[i-1].length()){
                minLenIndex = i;
            }
        }

        String minStr = strs[minLenIndex];

        int count = 0;
        int cc = minStr.length();
        for (int i = 0; i < cc; i++) {
            for (String str : strs) {
                if (str.startsWith(minStr)) {
                    count++;
                } else {
                    break;
                }
                if (count == strs.length) {
                    return minStr;
                }
            }
            minStr = minStr.substring(0, minStr.length() - 1);
            count = 0;
        }

        return minStr;
    }

    /**
     * 3.验证 IP地址
     *
     * 时间复杂度 O（n）  空间复杂度 O（n）
     * 验证IP地址
     * @param IP string字符串 一个IP地址字符串
     * @return string字符串
     */
    public String solve3(String IP) {
        if(isIPv4(IP))
            return "IPv4";
        else if(isIPv6(IP))
            return "IPv6";
        return "Neither";
    }
    boolean isIPv4 (String IP) {
        if(IP.indexOf('.') == -1){
            return false;
        }
        String[] s = IP.split("\\.");
        //IPv4必定为4组
        if(s.length != 4)
            return false;
        for(int i = 0; i < s.length; i++){
            //不可缺省，有一个分割为零，说明两个点相连
            if(s[i].length() == 0)
                return false;
            //比较数字位数及不为零时不能有前缀零
            if(s[i].length() < 0 || s[i].length() > 3 || (s[i].charAt(0)=='0' && s[i].length() != 1))
                return false;
            int num = 0;
            //遍历每个分割字符串，必须为数字
            for(int j = 0; j < s[i].length(); j++){
                char c = s[i].charAt(j);
                if (c < '0' || c > '9')
                    return false;
                //转化为数字比较，0-255之间
                num = num * 10 + (int)(c - '0');
                if(num < 0 || num > 255)
                    return false;
            }
        }
        return true;
    }
    boolean isIPv6 (String IP) {
        if (IP.indexOf(':') == -1) {
            return false;
        }
        String[] s = IP.split(":",-1);
        //IPv6必定为8组
        if(s.length != 8){
            return false;
        }
        for(int i = 0; i < s.length; i++){
            //每个分割不能缺省，不能超过4位
            if(s[i].length() == 0 || s[i].length() > 4){
                return false;
            }
            for(int j = 0; j < s[i].length(); j++){
                //不能出现a-fA-F以外的大小写字符
                char c = s[i].charAt(j);
                boolean expr = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') ;
                if(!expr){
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 4.大数加法
     *
     * 时间复杂度 O（n）   使用栈
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算两个数之和
     * @param s string字符串 表示第一个整数
     * @param t string字符串 表示第二个整数
     * @return string字符串
     */
    public String solve (String s, String t) {
        // write code here
        Stack<Integer> stack = new Stack<>();
        StringBuilder stringBuilder = new StringBuilder();
        int i = s.length() - 1, j = t.length() - 1, carry = 0;
        while (i >= 0 || j >= 0 || carry != 0) {
            carry += i >= 0 ? s.charAt(i--) - '0' : 0;
            carry += j >= 0 ? t.charAt(j--) - '0' : 0;
            stack.push(carry % 10); // 非进位
            carry = carry / 10;  // 进位
        }
        while (!stack.isEmpty())
            stringBuilder.append(stack.pop());
        return stringBuilder.toString();
    }

    /** 双指针
     * 1.合并两个有序数组
     *
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算两个数之和
     * @param A int数组
     * @param m int
     * @param B int数组
     * @param n int
     * @return void
     */
    public void merge(int A[], int m, int B[], int n) {
        int i = m - 1;
        int j = n - 1;
        int p = m + n - 1;


        while (i >= 0 && j >= 0){
            A[p--] = A[i] >= B[j]? A[i--]:B[j--];
        }

        while (j >= 0){
            A[p--] = B[j--];
        }
    }

    /**
     * 2.判断是否为回文字符串
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param str string字符串 待判断的字符串
     * @return bool布尔型
     */
    public boolean judge (String str) {
        // write code here

        int i = 0;
        int j = str.length() - 1;

        while (i < j){
            if (str.charAt(i) != str.charAt(j)){
                return false;
            }
            i++;
            j--;

        }
        return true;
    }

    /**
     * 3.合并区间
     *
     *
     * 时间复杂度：O（nlogn）  空间复杂度：O（n）
     * 进阶：时间复杂度 O（val）   空间复杂度 O（val）   val为区间里的值
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param intervals ArrayList数组
     * @return ArrayList数组
     */
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        if (intervals.size() == 0) return new ArrayList<>();

        // List排序
//        intervals.sort(new Comparator<Interval>() {
//            @Override
//            public int compare(Interval o1, Interval o2) {
//                if (o1.start != o2.start)
//                    return o1.start - o2.start;
//                else
//                    return o1.end - o2.end;
//            }
//        });

        // Array排序
        Interval[] intervals2 = intervals.toArray(new Interval[0]);
        Arrays.sort(intervals2, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                if (o1.start != o2.start)
                    return o1.start - o2.start;
                else
                    return o1.end - o2.end;
            }
        });
//        注意：需要按照上面分开写，合起来写排不了序
//        Arrays.sort(intervals.toArray(new Interval[0]), new Comparator<Interval>() {
//            @Override
//            public int compare(Interval o1, Interval o2) {
//                if (o1.start != o2.start)
//                    return o1.start - o2.start;
//                else
//                    return o1.end - o2.end;
//            }
//        });
//
//        for (Interval interval : intervals) {
//            System.out.println(interval.start + "   " + interval.end);
//        }

        ArrayList<Interval> res = new ArrayList<>();
        res.add(intervals2[0]);

        for (int i = 1; i < intervals2.length; i++) {
            if (intervals2[i].start <= res.get(res.size() - 1).end){ // 有重合
                int newEnd = Math.max(intervals2[i].end, res.get(res.size() - 1).end);
                System.out.println(newEnd);
                int newStart = res.get(res.size() - 1).start;
                System.out.println(newStart);
                res.remove(res.size() - 1);
                res.add(new Interval(newStart, newEnd));
            }else {
                res.add(intervals2[i]);
            }
        }

        return res;
    }

    /**
     *
     * 4.最长无重复子数组
     *
     * 要求：连续 且 不重复
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxLength (int[] arr) {
        // write code here
        // 哈希方法

        if (arr.length == 0) return 0;
        if (arr.length == 1) return 1;


        int res = 0;

        for (int i = 0; i < arr.length; i++) {
            HashSet<Integer> hashSet = new HashSet<>();
            hashSet.add(arr[i]);

            for (int j = i + 1; j < arr.length; j++) {
                int oldSize = hashSet.size();
                hashSet.add(arr[j]);
                int newSize = hashSet.size();
                if (oldSize == newSize){
                    res = Math.max(hashSet.size(), res);
                    break;
                }else {
                    res = Math.max(res, hashSet.size());
                }
            }
        }

        return res;
    }

    /**
     * 5.盛水最多的容器
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param height int整型一维数组
     * @return int整型
     */
    public int maxArea (int[] height) {
        // write code here
        if (height.length < 2){
            return 0;
        }

        int i = 0;
        int j = height.length - 1;
        int res = 0;
        while (i < j){
            int temp = Math.min(height[i], height[j]) * (j - i);
            res = Math.max(res, temp);
            if (height[i] < height[j]){
                i ++;
            }else if (height[i] > height[j]){
                j --;
            }else {
                i ++;
                j --;
            }
        }

        return res;
    }

    /**
     * 6.接雨水问题
     *
     * 时间复杂度：O（n）
     * max water
     * @param arr int整型一维数组 the array
     * @return long长整型
     */
    public long maxWater (int[] arr) {
        // write code here
        //排除空数组
        if(arr.length == 0)
            return 0;

        long res = 0;
        //左右双指针
        int left = 0;
        int right = arr.length - 1;

        //中间区域的边界高度
        int maxL = 0;
        int maxR = 0;

        //直到左右指针相遇
        while(left < right){
            //每次维护往中间的最大边界
            maxL = Math.max(maxL, arr[left]);
            maxR = Math.max(maxR, arr[right]);
            //较短的边界确定该格子的水量
            if(maxR > maxL)
                res += maxL - arr[left++];
            else
                res += maxR - arr[right--];
        }
        return res;
    }

    /**
     *
     * 7.最小覆盖子串
     *
     * 时间复杂度为 O（n）
     *
     * @return string字符串
     */
    //检查是否有小于0的
    boolean check(int[] hash) {
        for (int i = 0; i < hash.length; i++) {
            if (hash[i] < 0)
                return false;
        }
        return true;
    };

    public String minWindow (String S, String T) {
        int cnt = S.length() + 1;
        //记录目标字符串T的字符个数
        int[] hash = new int[128];
        for(int i = 0; i < T.length(); i++)
            //初始化哈希表都为负数，找的时候再加为正
            hash[T.charAt(i)] -= 1;
        int slow = 0, fast = 0;
        //记录左右区间
        int left = -1, right = -1;
        for(; fast < S.length(); fast++){
            char c = S.charAt(fast);
            //目标字符匹配+1
            hash[c]++;
            //没有小于0的说明都覆盖了，缩小窗口
            while(check(hash)){
                //取最优解
                if(cnt > fast - slow + 1){
                    cnt = fast - slow + 1;
                    left = slow;
                    right = fast;
                }
                c = S.charAt(slow);
                //缩小窗口的时候减1
                hash[c]--;
                //窗口缩小
                slow++;
            }
        }
        //找不到的情况
        if(left == -1)
            return "";
        return S.substring(left, right + 1);
    }

    /** 贪心
     * 1.主持人调度
     *
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算成功举办活动需要多少名主持人
     * @param n int整型 有n个活动
     * @param startEnd int整型二维数组 startEnd[i][0]用于表示第i个活动的开始时间，startEnd[i][1]表示第i个活动的结束时间
     * @return int整型
     */
    public int minmumNumberOfHost (int n, int[][] startEnd) {
        // write code here
        int[] start = new int[n];
        int[] end = new int[n];
        //分别得到活动起始时间
        for(int i = 0; i < n; i++){
            start[i] = startEnd[i][0];
            end[i] = startEnd[i][1];
        }
        //单独排序
        Arrays.sort(start, 0, start.length);
        Arrays.sort(end, 0, end.length);
        int res = 0;
        int j = 0;
        for(int i = 0; i < n; i++){
            //新开始的节目大于上一轮结束的时间，主持人不变
            if(start[i] >= end[j])
                j++;
            else
                //主持人增加
                res++;
        }
        return res;
    }

    /** 模拟
     *
     * 1.旋转数组
     *
     * 时间复杂度：O（n）  空间复杂度：O（1）
     * @param n int整型 数组长度
     * @param m int整型 右移距离
     * @param a int整型一维数组 给定数组
     * @return int整型一维数组
     */
    public int[] solve (int n, int m, int[] a) {
        //取余，因为每次长度为n的旋转数组相当于没有变化
        m = m % n;
        //第一次逆转全部数组元素
        reverse(a, 0, n - 1);
        //第二次只逆转开头m个
        reverse(a, 0, m - 1);
        //第三次只逆转结尾m个
        reverse(a, m, n - 1);
        return a;
    }
    //反转函数
    public void reverse(int[] nums, int start, int end){
        while(start < end){
            swap(nums, start++, end--);
        }
    }
    //交换函数
    public void swap(int[] nums, int a, int b){
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }

    /** 模拟
     *
     * 2.螺旋矩阵
     *
     * 时间复杂度：O（nm）  空间复杂度：O（nm）
     * @param matrix int[][]整型二维数组
     * @return ArrayList<Integer>
     */
    public ArrayList<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>();
        if(matrix.length == 0)
            return res;

        // 定义四个指针，并且充当边界限制的作用
        int top = 0, bottom = matrix.length-1;
        int left = 0, right = matrix[0].length-1;

        while( top < (matrix.length+1)/2 && left < (matrix[0].length+1)/2 ){
            //上面  左到右
            for(int i = left; i <= right; i++){
                res.add(matrix[top][i]);
            }

            //右边 上到下
            for(int i = top+1; i <= bottom; i++){
                res.add(matrix[i][right]);
            }

            //下面  右到左
            for(int i = right-1; top!=bottom && i>=left; i--){ // top !=bottom 避免只有一行的情况
                res.add(matrix[bottom][i]);
            }

            //左边 下到上
            for(int i = bottom-1; left!=right && i>=top+1; i--){ // left!=right 避免只有一列的情况
                res.add(matrix[i][left]);
            }
            // 遍历完一圈之后，所有往里面靠
            ++top;
            --bottom;
            ++left;
            --right;
        }
        return res;

    }

}

// 定义一个区间
class Interval{
     int start;
     int end;
     Interval() { start = 0; end = 0; }
     Interval(int s, int e) { start = s; end = e; }


}



// 二叉树节点
class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;
    }
}

// 单链表节点
class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}
