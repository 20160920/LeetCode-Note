package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

//21 合并
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	} else if l2 == nil {
		return l1
	} else if l1.Val < l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	} else {
		l2.Next = mergeTwoLists(l1, l2.Next)
		return l2
	}

}

// 35 插入数组位置
func searchInsert(nums []int, target int) int {
	var i int
	for i = 0; i < len(nums); i++ {
		if target == nums[i] {
			return i
		}
		if target < nums[i] {
			break
		}
	}
	return i
}

//53、最大子序和:给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
//1.暴力
func maxSubArray(nums []int) int {
	max := nums[0]
	temp := 0
	temp2 := 0
	if len(nums) == 1 {
		return nums[0]
	}
	for i := 0; i < len(nums); i++ {
		temp = nums[i]
		temp2 = nums[i]
		for j := i + 1; j < len(nums); j++ {
			temp2 += nums[j]
			if temp2 < nums[i] {
				continue
			}
			if temp2 > max {
				max = temp2
			}
		}
		if temp > max {
			max = temp
		}
	}
	return max
}

//58、最后一个单词的长度
//给你一个字符串 s，由若干单词组成，单词之间用空格隔开。返回字符串中最后一个单词的长度。如果不存在最后一个单词，请返回 0 。
//单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。
//func lengthOfLastWord(s string) int {
//	s1 := strings.Trim(s, " ")
//	return len(s1[strings.LastIndex(s1, " ")+1 :])
//}

//头 - 尾 从后往前第一处空格位置减去第二处空格
func lengthOfLastWord(s string) int {
	tail := len(s) - 1
	for tail >= 0 && s[tail] == 0 {
		tail--
	}
	if tail < 0 {
		return 0
	}
	head := tail
	for head >= 0 && s[head] != 0 {
		head--
	}
	return tail - head
}

//66. 加一
//给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
//最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
//你可以假设除了整数 0 之外，这个整数不会以零开头。

func climbStairs(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	if n == 3 {
		return 3
	}
	return climbStairs(n-1) + climbStairs(n-2)
}

//83. 删除排序链表中的重复元素
//存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次 。
//返回同样按升序排列的结果链表。

type ListNode struct {
	Val  int
	Next *ListNode
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	cur := head
	for cur.Next != nil {
		if cur.Val == cur.Next.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

//88
//给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
//初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。
//示例 1：
//输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
//输出：[1,2,2,3,5,6]
//示例 2：
//输入：nums1 = [1], m = 1, nums2 = [], n = 0
//输出：[1]
//func merge(nums1 []int, m int, nums2 []int, _ int) {
//	copy(nums1[m:],nums2)
//	sort.Ints(nums1)
//}

//func merge(nums1 []int, m int, nums2 []int, n int) {
//	sorted := make([]int, 0, m+n)
//	p1, p2 := 0, 0
//	for {
//		if p1 == m {
//			sorted = append(sorted, nums2[p2:]...)
//			break
//		}
//		if p2 == n {
//			sorted = append(sorted, nums1[p1:]...)
//			break
//		}
//		if nums1[p1] < nums2[p2] {
//			sorted = append(sorted, nums1[p1])
//			p1++
//		} else {
//			sorted = append(sorted, nums2[p2])
//			p2++
//		}
//	}
//	copy(nums1, sorted)
//}
//

//94. 二叉树的中序遍历
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//递归法
//func inorderTraversal(root *TreeNode) (res []int) {
//	var inorder func(node *TreeNode)
//	inorder = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		inorder(node.Left)
//		res = append(res, node.Val)
//		inorder(node.Right)
//	}
//	inorder(root)
//	return
//}

//迭代法
func inorderTraversal(root *TreeNode) (res []int) {
	stack := []*TreeNode{}
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, root.Val)
		root = root.Right
	}
	return res
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	var inorderTraversal func(root *TreeNode) (res []int, length int)
	inorderTraversal = func(root *TreeNode) (res []int, length int) {
		stack := []*TreeNode{}
		length = 0
		for root != nil || len(stack) > 0 {
			for root != nil {
				stack = append(stack, root)

				if root.Left == nil && root.Right != nil || root.Left != nil {
					length++
				}
				root = root.Left
			}
			root = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res = append(res, root.Val)

			root = root.Right
			if root != nil {
				length++
			}
		}
		return res, length
	}

	treeOne, lengthOne := inorderTraversal(p)
	treeTwo, lengthTwo := inorderTraversal(q)
	if len(treeOne) != len(treeTwo) || lengthOne != lengthTwo {
		return false
	}
	for i := 0; i < len(treeOne); i++ {
		if treeOne[i] == treeTwo[i] {
			continue
		} else {
			return false
		}
	}
	return true
}

//101.判断是否镜像二叉树
//递归
//func isSymmetric(root *TreeNode) bool {
//	return check(root, root)
//}
//
//func check(p,q *TreeNode) bool{
//	if p == nil && q == nil {
//		return true
//	}
//	if p == nil || q == nil {
//		return false
//	}
//	return p.Val == q.Val &&check(p.Right,q.Left) && check(p.Right,q.Left)
//}
//迭代
func isSymmetric(root *TreeNode) bool {
	u, v := root, root
	q := []*TreeNode{}
	q = append(q, u)
	q = append(q, v)
	for len(q) > 0 {
		u, v = q[0], q[1]
		q = q[2:]
		if u == nil && v == nil {
			continue
		}
		if u == nil || v == nil {
			return false
		}
		if u.Val != v.Val {
			return false
		}
		q = append(q, u.Left)
		q = append(q, v.Right)
		q = append(q, u.Right)
		q = append(q, v.Left)
	}
	return true
}

//104. 二叉树的最大深度
//给定一个二叉树，找出其最大深度。
//
//二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
//
//说明: 叶子节点是指没有子节点的节点。

//1.层次遍历
//
//func maxDepth(root *TreeNode) int {
//	q := [] *TreeNode{}
//	q = append(q, root)
//	height := 0
//	if root == nil {
//		return 0
//	}
//	for len(q) > 0 && root != nil{
//		width := len(q)
//		for i := 0; i < width; i++ {
//			if q[i].Left != nil {
//				q = append(q, q[i].Left)
//			}
//			if q[i].Right != nil{
//				q = append(q, q[i].Right)
//			}
//		}
//		q = q[width:]
//		if root.Left != nil {
//			root = root.Left
//		}
//		if root.Right != nil{
//			root = root.Right
//		}
//		height ++
//	}
//	return height
//}
//2.深度优先遍历
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}

//108. 将有序数组转换为二叉搜索树
//给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
//高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
func sortedArrayToBST(nums []int) *TreeNode {
	return helper(nums, 0, len(nums)-1)
}
func helper(nums []int, left, right int) *TreeNode {
	if left > right {
		return nil
	}

	mid := (left + right) / 2
	root := &TreeNode{Val: nums[mid]}
	root.Left = helper(nums, left, mid-1)
	root.Right = helper(nums, mid, right)
	return root
}

///111.二叉树最小深度
func minDepth(root *TreeNode) int {
	q := []*TreeNode{}
	q = append(q, root)
	height := 0
	if root == nil {
		return 0
	}
	for len(q) > 0 && root != nil {
		width := len(q)
		for i := 0; i < width; i++ {
			if q[i].Left != nil {
				q = append(q, q[i].Left)
			}
			if q[i].Right != nil {
				q = append(q, q[i].Right)
			}
			if q[i].Left == nil && q[i].Right == nil {
				return height + 1
			}
		}
		q = q[width:]
		if root.Left != nil {
			root = root.Left
		}
		if root.Right != nil {
			root = root.Right
		}
		height++
	}
	return height
}

//112. 路径总和
//给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。
//叶子节点 是指没有子节点的节点。
func hasPathSum(root *TreeNode, sum int) bool {
	if root == nil {
		return false
	}
	if root.Left == nil && root.Right == nil {
		return sum == root.Val
	}
	return hasPathSum(root.Left, sum-root.Val) || hasPathSum(root.Right, sum-root.Val)
}

//118.杨辉三角
//在杨辉三角中，每个数是它左上方和右上方的数的和。
//[
//      [1],
//     [1,1],
//    [1,2,1],
//   [1,3,3,1],
//  [1,4,6,4,1],
// [1,5,10,10,5,1],
//[1,6,15,20,15,6,1],
//]

func generate(numRows int) [][]int {
	res := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		res[i] = make([]int, i+1)
		for j := 0; j <= i; j++ {
			if j == 0 || j == i {
				res[i][j] = 1
			} else {
				res[i][j] = res[i-1][j-1] + res[i-1][j]
			}
		}
	}
	return res
}

func getRow(rowIndex int) []int {
	row := make([]int, rowIndex+1)
	row[0] = 1
	for i := 1; i <= rowIndex; i++ {
		for j := i; j > 0; j-- {
			row[j] += row[j-1]
		}
	}
	return row
}

//125. 验证回文串
//给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
//
//说明：本题中，我们将空字符串定义为有效的回文串。
//
//
//
//示例 1:
//
//输入: "A man, a plan, a canal: Panama"
//输出: true
//解释："amanaplanacanalpanama" 是回文串
//示例 2:
//
//输入: "race a car"
//输出: false
//解释："raceacar" 不是回文串
func isPalindrome(s string) bool {
	l, r := 0, len(s)-1
	s = strings.ToLower(s)
	for l < r {
		for l < r && !isalnum(s[l]) {
			l++
		}
		for l < r && !isalnum(s[r]) {
			r--
		}
		if l < r {
			if s[l] != s[r] {
				return false
			}
			l++
			r--
		}
	}
	return true
}

func isalnum(ch byte) bool {
	return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
}

//136. 只出现一次的数字
//给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
//说明：
//你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
//示例 1:
//输入: [2,2,1]
//输出: 1
//示例 2:
//输入: [4,1,2,1,2]
//输出: 4

//func singleNumber(nums []int) int {
//	sort.Ints(nums)
//	i := 0
//	for i < len(nums) {
//		if i+1 > len(nums)-1 || nums[i] != nums[i+1]&&nums[i] != nums[i+2] {
//			break
//		} else {
//			i = i + 3
//		}
//
//	}
//	return nums[i]
//}

//用异或 位运算
func singleNumber(nums []int) int {
	single := 0
	for _, num := range nums {
		single ^= num
	}
	return single
}

//141. 环形链表
//给定一个链表，判断链表中是否有环。
//如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	slow, fast := head, head.Next
	for fast != slow {
		if fast == nil || fast.Next == nil {
			return false
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return true
}

func detectCycle(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	slow, fast := head, head.Next
	for fast != slow {
		if fast == nil || fast.Next == nil {
			return nil
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return fast
}

func maximumTime(time string) string {
	t := []byte(time)
	if t[0] == '?' {
		if '4' <= t[1] && t[1] <= '9' {
			t[0] = '1'
		} else {
			t[0] = '2'
		}
	}
	if t[1] == '?' {
		if t[0] == '2' {
			t[1] = '3'
		} else {
			t[1] = '9'
		}
	}
	if t[3] == '?' {
		t[3] = '5'
	}
	if t[4] == '?' {
		t[4] = '9'
	}
	return string(t)
}

//144.二叉树前序遍历
func preorderTraversal(root *TreeNode) (res []int) {
	var preorder func(node *TreeNode)
	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		res = append(res, node.Val)
		preorder(node.Left)
		preorder(node.Right)
	}
	preorder(root)
	return
}

//226. 翻转二叉树
func invertTree(root *TreeNode) *TreeNode {
	var preorder func(node *TreeNode)
	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		temp := node.Left
		node.Left = node.Right
		node.Right = temp
		preorder(node.Left)
		preorder(node.Right)
	}
	return root
}

//257. 二叉树的所有路径
//给定一个二叉树，返回所有从根节点到叶子节点的路径。
var paths []string

func binaryTreePaths(root *TreeNode) []string {
	paths = []string{}
	constructPaths(root, "")
	return paths
}

func constructPaths(root *TreeNode, path string) {
	if root != nil {
		point := path
		point += strconv.Itoa(root.Val)
		if root.Left == nil && root.Right == nil {
			paths = append(paths, point)
		} else {
			point += "->"
			constructPaths(root.Left, point)
			constructPaths(root.Right, point)
		}

	}
}

//404. 左叶子之和
func isLeafNode(node *TreeNode) bool {
	return node.Left == nil && node.Right == nil
}

func dfs(node *TreeNode) (ans int) {
	if node.Left != nil {
		if isLeafNode(node.Left) {
			ans += node.Left.Val
		} else {
			ans += dfs(node.Left)
		}
	}
	if node.Right != nil && !isLeafNode(node.Right) {
		ans += dfs(node.Right)
	}
	return
}

func sumOfLeftLeaves(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return dfs(root)
}

//501. 二叉搜索树中的众数
//给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。
//假定 BST 有如下定义：
//结点左子树中所含结点的值小于等于当前结点的值
//结点右子树中所含结点的值大于等于当前结点的值
//左子树和右子树都是二叉搜索树
func findMode(root *TreeNode) (answer []int) {
	var base, count, maxCount int
	update := func(x int) {
		if x == base {
			count++
		} else {
			base, count = x, 1
		}
		if count > maxCount {
			maxCount = count
			answer = []int{base}
		} else if count == maxCount {
			answer = append(answer, base)
		}
	}
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		update(node.Val)
		dfs(node.Right)
	}
	dfs(root)
	return
}

//530. 二叉搜索树的最小绝对差
//给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。
func getMinimumDifference(root *TreeNode) int {
	pre := -1
	minDiff := math.MaxInt64
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if pre != -1 && node.Val-pre < minDiff {
			minDiff = node.Val - pre
		}
		pre = node.Val
		dfs(node.Right)
	}
	dfs(root)
	return minDiff
}

//543. 二叉树的直径
// 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
func diameterOfBinaryTree(root *TreeNode) int {
	var maxDiameter int
	var dfs func(node *TreeNode) int

	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		} else {
			leftDeep := dfs(node.Left)
			rightDeep := dfs(node.Right)
			if leftDeep+rightDeep > maxDiameter {
				maxDiameter = leftDeep + rightDeep
			}
			return max(leftDeep, rightDeep) + 1
		}
	}
	dfs(root)
	return maxDiameter
}

//563. 二叉树的坡度
//给定一个二叉树，计算 整个树 的坡度 。
//一个树的 节点的坡度 定义即为，该节点左子树的节点之和和右子树节点之和的 差的绝对值 。如果没有左子树的话，左子树的节点之和为 0 ；没有右子树的话也是一样。空结点的坡度是 0 。
//整个树 的坡度就是其所有节点的坡度之和。
func findTilt(root *TreeNode) int {

	tilt := 0
	var traverse func(node *TreeNode) int
	traverse = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := traverse(node.Left)
		right := traverse(node.Right)
		tilt += int(math.Abs(float64(left - right)))
		return left + right + node.Val
	}
	traverse(root)
	return tilt
}

//606. 根据二叉树创建字符串
//你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。
//空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树之间的一对一映射关系的空括号对。

func tree2str(t *TreeNode) string {
	var str strings.Builder
	if t == nil {
		return ""
	}
	stack := []interface{}{t}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if n, ok := node.(*TreeNode); ok {
			str.WriteString(fmt.Sprintf("%d", n.Val))
			if n.Right != nil {
				stack = append(stack, ")")
				stack = append(stack, n.Right)
				stack = append(stack, "(")
			}
			if n.Right != nil && n.Left == nil {
				stack = append(stack, "()")
			}
			if n.Left != nil {
				stack = append(stack, ")")
				stack = append(stack, n.Left)
				stack = append(stack, "(")
			}
		} else {
			s := node.(string)
			str.WriteString(s)
		}
	}
	return str.String()
}

//617. 合并二叉树
//给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
//你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
//func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
//
//}

//338. 比特位计数
//给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
//示例 1:
//输入: 2
//输出: [0,1,1]
//示例 2:
//输入: 5
//输出: [0,1,1,2,1,2]

//Brian Kernighan 算法
func countBits(n int) []int {
	res := make([]int, n+1)
	var oneCount func(x int) int
	oneCount = func(x int) int {
		one := 0
		for ; x > 0; x &= x - 1 {
			one++
		}
		return one
	}
	for i := range res {
		res[i] = oneCount(i)
	}
	return res
}

//动态规划+位运算
func countBits2(n int) []int {
	bits := make([]int, n+1)
	for i := 1; i <= n; i++ {
		bits[i] = bits[i>>1] + i&1
	}
	return bits
}

//
//func isSubsequence(s string, t string) bool {
//	if len(s) > len(t) {
//		return false
//	}
//	x := 0
//	for i := 0; i < len(t); i++ {
//		if s[x] == t[i]-1
//			x++
//		}
//		if x == len(s) {
//			return true
//		}
//	}
//	return false
//}

//322. 零钱兑换
func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func min3(a, b, c int) int {
	if a < b && a < c {
		return a
	} else if b < a && b < c {
		return b
	}
	return c
}
func coinChange(coins []int, amount int) int {
	//dp数组的定义：凑出总金额amount,至少需要dp[amount]枚硬币
	dp := make([]int, amount+1)
	//base case
	dp[0] = 0
	//外层for遍历所有状态的所有取值
	for i := 1; i < len(dp); i++ {
		// dp 数组全部初始化为特殊值 amount + 1，也就是大于总目标
		dp[i] = amount + 1
		//内层for循环求所有选择的最小值
		for _, coin := range coins {
			if i-coin < 0 {
				continue
			}
			dp[i] = min2(dp[i], 1+dp[i-coin])
		}
	}
	if dp[amount] == amount+1 {
		return -1
	}
	return dp[amount]
}

//72. 编辑距离
//给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
//你可以对一个单词进行如下三种操作：
//插入一个字符
//删除一个字符
//替换一个字符
//
//func minDistance(word1 string, word2 string) int {
//	m := len(word1)
//	n := len(word2)
//	dp := make([][]int, m+1)
//
//	for i := range dp {
//		dp[i] = make([]int, n+1)
//	}
//
//	//base case
//	for i := 0; i < m+1; i++ {
//		dp[i][0] = i
//	}
//	for j := 0; j < n+1; j++ {
//		dp[0][j] = j
//	}
//	//自底向上求解
//	for i := 1; i <= m; i++ {
//		for j := 1; j <= n; j++ {
//			if word1[i-1] == word2[j-1] {
//				dp[i][j] = dp[i-1][j-1]
//			} else {
//				dp[i][j] = Min(
//					dp[i-1][j]+1,
//					dp[i][j-1]+1,
//					dp[i-1][j-1]+1)
//			}
//		}
//	}
//	return dp[m][n]
//}
//
//
//
////931. 下降路径最小和
////给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。
////下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。
////在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。
////具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
//func minFallingPathSum(matrix [][]int) int {
//	n := len(matrix)
//	for i := 0; i <= n; i++ {
//		for j := 0; j <= n; j++ {
//			minN := matrix[i+1][j]
//			if j > 0 {
//				minN = Min(minN,matrix[i+1][j-1])
//			}
//			if j+1 < n {
//				minN = Min(minN,matrix[i+1][j+1])
//			}
//		matrix[i][j] += minN + matrix[i][j]
//		}
//	}
//	res := matrix[n-1][0]
//	for j := 1; j < n; j++ {
//		res = Min(res, matrix[n-1][j])
//	}
//	return  res
//}
var res []string
var phoneMap = map[string]string{
	"1": "abc",
	"2": "def",
	"3": "ghi",
	"4": "jkl",
	"5": "mno",
	"6": "pqrs",
	"7": "tuv",
	"8": "wxyz",
}

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	res = []string{}
	backtrack(digits, 0, "")
	return res
}
func backtrack(digits string, index int, temp string) {
	if index == len(digits) {
		res = append(res, temp)
	} else {
		digit := string(digits[index])
		letters := phoneMap[digit]
		length := len(letters)
		for i := 0; i < length; i++ {
			backtrack(digits, index+1, temp+string(letters[i]))
		}
	}
}

func main() {
	s := string(5 + 'A')
	println(s)

}
